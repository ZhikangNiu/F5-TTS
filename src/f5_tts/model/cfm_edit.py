"""
CFMEdit — CFM variant that uses a pre-trained LLM (Qwen2.5-Omni) as text encoder.

Inherits from CFM and overrides text processing:
- Training (forward): encodes text strings via LLM, passes hidden states to backbone
- Inference (sample): encodes text once outside ODE loop

ein notation:
b - batch
n - sequence
nt - text sequence
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import logging
from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.cfm import CFM
from f5_tts.model.utils import exists, get_epss_timesteps, lens_to_mask, mask_from_frac_lengths


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = get_logger(__name__)


class CFMEdit(CFM):
    def __init__(
        self,
        transformer,
        text_encoder,
        text_processor,
        text_encoder_max_length: int = 512,
        text_drop_idx: int = 40,  # ignore template token
        **kwargs,
    ):
        # CFM.__init__ expects vocab_char_map; we don't use it, but pass None
        kwargs.setdefault("vocab_char_map", None)
        super().__init__(transformer, **kwargs)

        self.text_encoder = text_encoder
        self.text_processor = text_processor
        self.text_encoder_max_length = text_encoder_max_length
        self.text_drop_idx = text_drop_idx
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        self.system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

    @torch.no_grad()
    def encode_text(self, text: list[str], device) -> torch.Tensor:
        """Encode text strings via LLM into hidden states [B, seq, hidden_dim]."""

        # 构建 chat messages
        messages_batch = [
            [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": t}]},
            ]
            for t in text
        ]

        # 用 apply_chat_template 格式化为 chat 文本
        formatted_texts = self.text_processor.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=True,
        )

        # tokenize
        tokens = self.text_processor(
            text=formatted_texts,
            padding=True,
            padding_side="right",  # defaults left padding
            truncation=True,
            max_length=self.text_encoder_max_length,
            return_tensors="pt",
        ).to(device)

        # ensure text encoder stays in eval mode even if model is in train mode
        self.text_encoder.eval()

        outputs = self.text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [B, seq, hidden_dim]
        return hidden[:, self.text_drop_idx :, :], tokens.attention_mask[:, self.text_drop_idx :].bool()

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: list[str] | float["b nt h"],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()

        # raw wave -> mel
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)
        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # encode text (once, outside ODE loop)
        text_embeds, context_mask = self.encode_text(text, device)

        # duration
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        # duration at least text/audio prompt length plus one token
        non_pad = (text_embeds.abs().sum(-1) > 0).sum(dim=-1)  # valid text tokens
        duration = torch.maximum(torch.maximum(non_pad, lens) + 1, duration)
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        # ODE fn
        def fn(t, x):
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text_embeds,
                    time=t,
                    mask=mask,
                    c_mask=context_mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text_embeds,
                time=t,
                mask=mask,
                c_mask=context_mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        # Always compute timesteps in fp32 to avoid bf16 precision collapse
        # (e.g. cos(small_x) rounds to 1.0 in bf16, making adjacent steps equal)
        if t_start == 0 and use_epss:
            t = get_epss_timesteps(steps, device=self.device, dtype=torch.float32)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=torch.float32)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],
        text: list[str] | float["b nt h"],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # encode text via LLM

        text_embeds, context_mask = self.encode_text(text, device)

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # random span mask for infilling training
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp
        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = self.sample_time(batch, dtype, self.device)

        # sample xt
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict within the random mask span
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob
        if random() < self.cond_drop_prob:
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        pred = self.transformer(
            x=φ,
            cond=cond,
            text=text_embeds,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,
            c_mask=context_mask,
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
