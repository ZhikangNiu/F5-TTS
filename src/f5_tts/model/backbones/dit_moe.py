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
import logging
from x_transformers.x_transformers import RotaryEmbedding

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    DiTMoEBlock,
    AdaLayerNormZero_Final,
    AdaRMSNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)




class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        return y

# Text embedding
class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2, refine_type="conv"):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token
        self.refine_type = refine_type
        if refine_type == "conv":
            if conv_layers > 0:
                self.extra_modeling = True
                self.precompute_max_pos = 4096  # ~44s of 24khz audio
                self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
                self.text_blocks = nn.Sequential(
                    *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
                )
            else:
                self.extra_modeling = False
        elif refine_type == "conv_bilstm":
            if conv_layers > 0:
                self.extra_modeling = True
                self.precompute_max_pos = 4096  # ~44s of 24khz audio
                self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
                self.text_blocks = nn.Sequential(
                    *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)],
                    SLSTM(dimension=text_dim,num_layers=2,skip=False)
                )
            else:
                self.extra_modeling = False
        elif refine_type == "transformer_encoder":
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            text_layer = nn.TransformerEncoderLayer(d_model=text_dim,nhead=8,batch_first=True)
            self.text_blocks = nn.TransformerEncoder(text_layer,num_layers=conv_layers)
            
    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)
        src_key_padding_mask = (text == 0) # shape: [batch_size, seq_len], True 表示需要被 mask 的位置

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.refine_type in ["conv_bilstm","conv"]:
                for layer in self.text_blocks:
                    text = layer(text)
                    text = text.masked_fill(src_key_padding_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)),0.0)
            elif self.refine_type == "transformer_encoder":
                text = self.text_blocks(text,src_key_padding_mask=src_key_padding_mask)
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

class DiTMoE(nn.Module):
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
        refine_type="conv",
        norm_type="ln",
        silu_ff=False,
        zero_init=False,
        qk_norm=False,
        num_experts=16,
        num_experts_per_tok=2,
        pretraining_tp=2
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers,refine_type=refine_type)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth
        self.norm_type = norm_type
        if self.norm_type == "ln":
            logger.info("Use Layer Norm")
            self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        elif self.norm_type == "rmsnorm":
            logger.info("Use RMSNorm")
            self.norm_out = AdaRMSNormZero_Final(dim)
        if qk_norm:
            logger.info("Attn use qk_norm")
        
        #  dim, heads, dim_head, ff_mult=4, dropout=0.1,norm_type="ln",num_experts=16, num_experts_per_tok=2, pretraining_tp=2
        self.transformer_blocks = nn.ModuleList(
            [DiTMoEBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout,norm_type=self.norm_type,num_experts=num_experts) for _ in range(depth)]
        )

        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations
        
        if zero_init:
            logger.info("Zero Init Model")
            self.initialize_weights()

    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)
        
    
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
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
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
