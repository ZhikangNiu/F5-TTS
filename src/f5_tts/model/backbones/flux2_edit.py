"""
Flux2Edit backbone — Flux2Audio variant using pre-encoded LLM text embeddings.

Two-phase architecture:
  1. Double-stream MMDiTBlock (joint text-audio attention)
  2. Single-stream DiTBlock (concatenated text+audio)

Replaces TextEmbedding with RMSNorm + Linear projection for Qwen2.5-Omni LLM.

ein notation:
b - batch
n - sequence
nt - text sequence
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.backbones.mmdit import AudioEmbedding
from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    DiTBlock,
    MMDiTBlock,
    RMSNorm,
    TimestepEmbedding,
)


class Flux2Edit(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_hidden_dim=2048,
        qk_norm=None,
        checkpoint_activations=False,
        attn_backend="torch",
        attn_mask_enabled=False,
        ffn_type="gelu",
        ffn_type_x=None,
        ffn_type_c=None,
        num_layers=8,
        num_single_layers=24,
        **kwargs,
    ):
        super().__init__()

        ffn_type_x = ffn_type_x or ffn_type
        ffn_type_c = ffn_type_c or ffn_type

        self.dim = dim
        self.depth = depth

        self.time_embed = TimestepEmbedding(dim)

        # text projection: RMSNorm(text_hidden_dim) -> Linear(text_hidden_dim, dim)
        self.txt_norm = RMSNorm(text_hidden_dim, eps=1e-6)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)

        self.audio_embed = AudioEmbedding(mel_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        # Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    qk_norm=qk_norm,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                    ffn_type_x=ffn_type_x,
                    ffn_type_c=ffn_type_c,
                )
                for i in range(num_layers)
            ]
        )

        # Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                    ffn_type=ffn_type,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        # text cache (mirrors Flux2Audio interface)
        self.text_cond, self.text_uncond = None, None

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)

        for block in self.single_transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)

        return ckpt_forward

    def project_text(self, text: float["b nt h"], drop_text: bool = False):
        """RMSNorm + Linear projection from LLM hidden states to model dim."""
        c = self.txt_proj(self.txt_norm(text))
        if drop_text:
            c = torch.zeros_like(c)
        return c

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b n d"],  # noised input audio
        cond: float["b n d"],  # masked cond audio
        text: float["b nt h"],  # pre-encoded text embeddings from LLM
        time: float["b"] | float[""],  # time step
        mask: bool["b n"] | None = None,
        c_mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ):
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        text_len = text.shape[1]

        # 这里得看下mask的部分
        # text mask: padding positions are all-zero in LLM output
        if c_mask is None:
            c_mask = text.abs().sum(-1) > 0  # [B, nt], True = valid

        if cfg_infer:
            # cond branch
            if cache and self.text_cond is not None:
                c_cond = self.text_cond
            else:
                c_cond = self.project_text(text, drop_text=False)
                if cache:
                    self.text_cond = c_cond
            x_cond = self.audio_embed(x, cond, drop_audio_cond=False)

            # uncond branch
            if cache and self.text_uncond is not None:
                c_uncond = self.text_uncond
            else:
                c_uncond = self.project_text(text, drop_text=True)
                if cache:
                    self.text_uncond = c_uncond
            x_uncond = self.audio_embed(x, cond, drop_audio_cond=True)

            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            c_mask = torch.cat((c_mask, c_mask), dim=0) if c_mask is not None else None
        else:
            c = self.project_text(text, drop_text=drop_text)
            x = self.audio_embed(x, cond, drop_audio_cond=drop_audio_cond)

        seq_len = x.shape[1]
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)

        # Phase 1: Double-stream MMDiTBlock
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                c, x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, c, t, mask, rope_audio, rope_text, c_mask, use_reentrant=False
                )
            else:
                c, x = block(x, c, t, mask=mask, rope=rope_audio, c_rope=rope_text, c_mask=c_mask)

        # Phase 2: Concatenate text+audio -> single-stream DiTBlock
        x = torch.cat([c, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(seq_len + text_len)

        if mask is not None:
            mask = torch.cat([c_mask, mask], dim=1)

        for block in self.single_transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        # extract audio portion
        x = x[:, text_len:]
        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
