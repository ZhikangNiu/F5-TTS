# training script for edit models (CFMEdit + MMDiTEdit / Flux2Edit)

import os
import shutil


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from importlib.resources import files
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from f5_tts.model import CFMEdit, Trainer
from f5_tts.model.dataset import load_dataset


os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    # Copy source code to Hydra output dir for reproducibility (rank 0 only)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        src_dir = Path(str(files("f5_tts").joinpath(".")))
        shutil.copytree(src_dir, output_dir / "f5_tts", dirs_exist_ok=True)

    # resolve backbone class
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = OmegaConf.to_container(model_cfg.model.arch, resolve=True)
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    wandb_project = model_cfg.ckpts.get("wandb_project", "CFM-TTS")
    wandb_run_name = model_cfg.ckpts.get(
        "wandb_run_name",
        f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.datasets.name}",
    )
    wandb_resume_id = model_cfg.ckpts.get("wandb_resume_id", None)

    # --- load text encoder by name ---
    WEIGHT_TYPE_MAP = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    text_encoder_cfg = model_cfg.model.text_encoder
    weight_dtype = WEIGHT_TYPE_MAP[text_encoder_cfg.get("weight_type", "bf16")]

    if text_encoder_cfg.name == "qwen_omni_3b":
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            text_encoder_cfg.text_encoder_path,
            torch_dtype=weight_dtype,
        )
        text_encoder = thinker.model  # Qwen2_5OmniThinkerTextModel
        del thinker  # release audio_tower + visual parts
        text_processor = Qwen2_5OmniProcessor.from_pretrained(text_encoder_cfg.text_encoder_path)
    else:
        raise ValueError(f"Unknown text encoder: {text_encoder_cfg.name}")

    # --- create backbone ---
    backbone = model_cls(
        **model_arc,
        mel_dim=model_cfg.model.mel_spec.n_mel_channels,
        text_proj_first=text_encoder_cfg.get("text_proj_first", False),
    )

    # --- create CFMEdit ---
    schedule_cfg = OmegaConf.to_container(model_cfg.model.get("schedule", OmegaConf.create({})), resolve=True)
    model = CFMEdit(
        transformer=backbone,
        text_encoder=text_encoder,
        text_processor=text_processor,
        text_encoder_max_length=text_encoder_cfg.text_encoder_max_length,
        text_drop_idx=text_encoder_cfg.text_drop_idx,
        mel_spec_kwargs=OmegaConf.to_container(model_cfg.model.mel_spec, resolve=True),
        **schedule_cfg,
    )

    # --- init trainer ---
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        fused_adam=model_cfg.optim.fused_adam,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    train_dataset = load_dataset(
        model_cfg.datasets.name, model_cfg.model.tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec
    )
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()
