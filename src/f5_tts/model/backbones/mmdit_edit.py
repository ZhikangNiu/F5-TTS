"""
MMDiTEdit backbone — MMDiT variant using pre-encoded LLM text embeddings.

Replaces TextEmbedding (nn.Embedding + ConvNeXt) with RMSNorm + Linear projection,
designed for use with Qwen2.5-Omni LLM as text encoder.

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
    MMDiTBlock,
    RMSNorm,
    TimestepEmbedding,
)


class MMDiTEdit(nn.Module):
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
        text_proj_first=False,
        qk_norm=None,
        checkpoint_activations=False,
        attn_backend="torch",
        attn_mask_enabled=False,
        ffn_type="gelu",
        ffn_type_x=None,
        ffn_type_c=None,
        # kept for hydra compat but unused
        text_num_embeds=256,
        text_mask_padding=True,
        text_layers=0,
        text_mult=2,
        text_block_type="conv",
        text_dim=None,
        **kwargs,
    ):
        super().__init__()

        ffn_type_x = ffn_type_x or ffn_type
        ffn_type_c = ffn_type_c or ffn_type

        self.dim = dim
        self.depth = depth

        self.time_embed = TimestepEmbedding(dim)
        self.text_proj_first = text_proj_first

        # text projection
        # False (default): RMSNorm(text_hidden_dim) -> Linear(text_hidden_dim, dim)
        # True:            Linear(text_hidden_dim, dim) -> RMSNorm(dim)
        norm_dim = dim if text_proj_first else text_hidden_dim
        self.txt_norm = RMSNorm(norm_dim, eps=1e-6)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)

        self.audio_embed = AudioEmbedding(mel_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    context_pre_only=i == depth - 1,
                    qk_norm=qk_norm,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                    ffn_type_x=ffn_type_x,
                    ffn_type_c=ffn_type_c,
                )
                for i in range(depth)
            ]
        )
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        # text cache (mirrors MMDiT interface)
        self.text_cond, self.text_uncond = None, None

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)

        return ckpt_forward

    def _project_text(self, text: float["b nt h"], drop_text: bool = False):
        """Project LLM hidden states to model dim with RMSNorm."""
        if self.text_proj_first:
            c = self.txt_norm(self.txt_proj(text))
        else:
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
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ):
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)

        # text mask: padding positions are all-zero in LLM output
        c_mask = text.abs().sum(-1) > 0  # [B, nt], True = valid

        if cfg_infer:
            # cond branch
            if cache and self.text_cond is not None:
                c_cond = self.text_cond
            else:
                c_cond = self._project_text(text, drop_text=False)
                if cache:
                    self.text_cond = c_cond
            x_cond = self.audio_embed(x, cond, drop_audio_cond=False)

            # uncond branch
            if cache and self.text_uncond is not None:
                c_uncond = self.text_uncond
            else:
                c_uncond = self._project_text(text, drop_text=True)
                if cache:
                    self.text_uncond = c_uncond
            x_uncond = self.audio_embed(x, cond, drop_audio_cond=True)

            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            c_mask = torch.cat((c_mask, c_mask), dim=0)
        else:
            c = self._project_text(text, drop_text=drop_text)
            x = self.audio_embed(x, cond, drop_audio_cond=drop_audio_cond)

        seq_len = x.shape[1]
        text_len = text.shape[1]
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                c, x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, c, t, mask, rope_audio, rope_text, c_mask, use_reentrant=False
                )
            else:
                c, x = block(x, c, t, mask=mask, rope=rope_audio, c_rope=rope_text, c_mask=c_mask)

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
