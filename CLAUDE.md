# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F5-TTS is a text-to-speech system using **Conditional Flow Matching** with Diffusion Transformers. It generates speech by conditioning on a reference audio clip and synthesizing new speech via ODE-based sampling. The project provides two model variants:
- **F5-TTS (v1)**: DiT backbone with ConvNeXt V2 text encoding — the primary model
- **E2-TTS**: UNetT backbone (flat-UNet Transformer reproduction)

Audio representation: mel-spectrograms (100 channels, 24kHz, hop_length=256), reconstructed to waveforms via Vocos or BigVGAN vocoder.

## Common Commands

### Installation (editable, for development/training)
```bash
pip install -e .
```

### Inference
```bash
f5-tts_infer-cli --ref_audio ref.wav --ref_text "..." --gen_text "..."
f5-tts_infer-gradio   # launches web UI
```

### Training
```bash
accelerate config                          # configure multi-GPU, mixed precision
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml

# override config values via Hydra
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml \
  ++datasets.batch_size_per_gpu=19200
```

### Finetuning
```bash
f5-tts_finetune-cli
f5-tts_finetune-gradio   # Gradio UI for finetuning
```

### Dataset Preparation
```bash
# Specific datasets (fill paths in scripts first)
python src/f5_tts/train/datasets/prepare_emilia.py
python src/f5_tts/train/datasets/prepare_libritts.py
python src/f5_tts/train/datasets/prepare_ljspeech.py

# Custom CSV dataset (format: audio_file|text, absolute paths)
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /path/to/output
```

### Linting
```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

Ruff config: line-length 120, target Python 3.10+, imports sorted after 2 blank lines.

## Code Style & Verification

Every code change must pass the repository's pre-commit checks before committing. After modifying code, always run:
```bash
pre-commit run --all-files
```

Key rules enforced by Ruff (v0.11.2):
- Line length: 120 characters max
- Target: Python 3.10+
- Imports: auto-sorted, 2 blank lines after import block
- Auto-fix enabled for lint issues and formatting

If pre-commit is not yet installed locally:
```bash
pip install pre-commit && pre-commit install
```

## Architecture

### Core Pipeline

```
Text → Tokenizer (pinyin/char) → Text Embedding + ConvNeXt V2
                                          ↓
Reference Audio → Mel Spectrogram → CFM (Conditional Flow Matching) → ODE Solver → Generated Mel → Vocoder → Waveform
```

### Key Source Layout (`src/f5_tts/`)

- **`model/cfm.py`** — `CFM` class: the central model wrapping a transformer backbone with flow matching. `forward()` computes training loss (MSE between predicted and true flow). `sample()` runs ODE-based inference with classifier-free guidance and sway sampling.
- **`model/backbones/dit.py`** — `DiT`: Diffusion Transformer, the default F5-TTS v1 backbone (dim=1024, depth=22, heads=16).
- **`model/backbones/mmdit.py`** — `MMDiT`: Multimodal DiT variant.
- **`model/backbones/unett.py`** — `UNetT`: flat-UNet Transformer for E2-TTS.
- **`model/modules.py`** — Building blocks: MelSpec, audio/text embeddings, attention, ConvNeXtV2 blocks.
- **`model/dataset.py`** — `HFDataset`, `CustomDataset`, and `load_dataset` for data loading.
- **`model/trainer.py`** — Training loop using HF Accelerate with EMA, gradient accumulation, W&B/TensorBoard logging.
- **`model/utils.py`** — Tokenizer loading (`get_tokenizer`), masking, positional embeddings.
- **`infer/utils_infer.py`** — Inference utilities: model/vocoder loading, audio preprocessing, the `infer_process` pipeline.
- **`api.py`** — `F5TTS` class: high-level Python API for inference.
- **`configs/`** — Hydra YAML configs (e.g., `F5TTS_v1_Base.yaml`). Training uses `hydra.main()` with config path resolution.

### How Training Works

1. `train/train.py` loads a Hydra config, instantiates the backbone class dynamically via `hydra.utils.get_class(f"f5_tts.model.{cfg.model.backbone}")`.
2. A `CFM` model wraps the backbone transformer.
3. `Trainer` handles distributed training via Accelerate, EMA updates, checkpoint saving (every 50k updates by default).
4. Loss: MSE between predicted flow and ground-truth flow (`x1 - x0`), computed only within randomly masked spans (infilling objective).

### How Inference Works

1. Reference audio is clipped to ~12s, resampled to 24kHz, converted to mel-spectrogram.
2. Text is normalized and tokenized (pinyin for Chinese, character-based otherwise).
3. CFM `sample()` runs an ODE from noise → mel-spectrogram using Euler method (default 32 steps), with classifier-free guidance and optional sway sampling.
4. Vocoder (Vocos by default) reconstructs the waveform from mel-spectrogram.

### Configuration System

Training configs live in `src/f5_tts/configs/` as Hydra YAML files. Key config groups:
- `datasets`: name, batch_size_per_gpu (frame-based), num_workers
- `model`: backbone (DiT/MMDiT/UNetT), arch (dim, depth, heads), mel_spec params, tokenizer
- `optim`: learning_rate, warmup, gradient clipping
- `ckpts`: logger type, save frequency, checkpoint retention

Override any value at CLI: `++model.arch.depth=24`.

### Local Training Setup

The `run.sh` script demonstrates multi-GPU training with a local vocoder path and W&B offline mode. Pass the config name as the first argument: `bash run.sh F5TTS_v1_Base.yaml`.
