"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    MMDiTBlock,
    TextTransformerBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)
from f5_tts.model.backbones.mmdit import TextEmbedding, AudioEmbedding

class Flux2Audio(nn.Module):
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
        text_num_embeds=256,
        text_mask_padding=True,
        qk_norm=None,
        text_layers=0,
        text_mult=2,
        text_block_type="conv",
        text_dim=None,
        checkpoint_activations=False,
        attn_backend="torch",
        attn_mask_enabled=False,
        ffn_type="gelu",
        num_layers=8,
        num_single_layers=24
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(
            dim,
            text_num_embeds,
            mask_padding=text_mask_padding,
            text_layers=text_layers,
            text_mult=text_mult,
            text_block_type=text_block_type,
            text_dim=text_dim,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.audio_embed = AudioEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

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
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
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
        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in MMDiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)

        for block in self.single_transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        text,  # b nt
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
    ):
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, drop_text=True)
                c = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, drop_text=False)
                c = self.text_cond
        else:
            c = self.text_embed(text, drop_text=drop_text)
        x = self.audio_embed(x, cond, drop_audio_cond=drop_audio_cond)

        return x, c

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b n d"],  # nosied input audio
        cond: float["b n d"],  # masked cond audio
        text: int["b nt"],  # text
        time: float["b"] | float[""],  # time step
        mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
    ):
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning (time), c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        c_mask = (text + 1) != 0  # True = valid, False = padding (-1 tokens)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond, c_cond = self.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache)
            x_uncond, c_uncond = self.get_input_embed(x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache)
            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            c_mask = torch.cat((c_mask, c_mask), dim=0)
        else:
            x, c = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache
            )

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
        
        # concat hidden states in seq dim
        x = torch.cat([c, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(seq_len + text_len)

        # extend mask to cover both text and audio tokens
        if mask is not None:
            mask = torch.cat([c_mask, mask], dim=1)

        for block in self.single_transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False
                )
            else:
                x = block(x, t, mask=mask, rope=rope)
        
        # get audio output
        x = x[:, text_len:]
        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
