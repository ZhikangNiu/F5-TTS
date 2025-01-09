"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


# Language ID embedding
class LanguageIDEmbedding(nn.Module):
    def __init__(self, lang_nums,out_dim, padding):
        super().__init__()
        self.language_embedding = nn.Embedding(lang_nums,out_dim)
        self.padding = padding

    def forward(self, x: torch.LongTensor):  # noqa: F722
        language_id_embedding = self.language_embedding(x)
        return language_id_embedding
# Text embedding

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2,lang_id_module=None):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token
        self.lang_embed = lang_id_module

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, lang, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0) # a b c -> a b c 0 0 0 0 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)
            lang = torch.zeros_like(lang)
        
        text = self.text_embed(text.long())  # b n -> b n d
        
        if self.lang_embed is not None:
            if self.lang_embed.padding:
                lang = self.lang_embed(lang).expand(-1, text_len, -1) # lang+text, lang+text , lang+text, lang+text, lang+text, pad, pad,pad
                lang = F.pad(lang, (0, 0, 0,seq_len - text_len), value=0) # same as text padding 需要check这里
            else:
                lang = self.lang_embed(lang).expand(-1,text.shape[1],-1) # 不做padding，这样避免了一句话前面是同一种语言，后面切换成另一种语言
            text = text + lang

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos) # 为什么pos embedding还要对padding的部分做
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
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
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_lang_id = False,
        lang_nums = 0,
        lang_padding = False
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.lang_id_embed = None
        if text_dim is None:
            text_dim = mel_dim
            
        if lang_nums > 0 and use_lang_id:
            self.lang_id_embed = LanguageIDEmbedding(lang_nums,text_dim,lang_padding=lang_padding) # 47 1 512
        
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers,lang_id_module=self.lang_id_embed)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        langs: float["b 1"],
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(langs, text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
