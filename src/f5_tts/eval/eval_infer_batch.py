import os
import sys

sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
from f5_tts.eval.utils_eval import (
    get_inference_prompt,
    get_latent_inference_prompt,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
)
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM, DiT, UNetT
from f5_tts.model.utils import get_tokenizer

accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"


# --------------------- Dataset Settings -------------------- #

target_sample_rate = 24000
hop_length = 256
win_length = 1024
n_fft = 1024
target_rms = 0.1

rel_path = str(files("f5_tts").joinpath("../../"))


def main():
    # ---------------------- infer setting ---------------------- #

    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-s", "--seed", default=None, type=int)
    parser.add_argument("-d", "--dataset", default="Emilia_ZH_EN")
    parser.add_argument("-n", "--expname", required=True)
    parser.add_argument("-c", "--ckptstep", default=1200000, type=int)
    parser.add_argument("-m", "--mel_spec_type", default="vocos", type=str, choices=["bigvgan", "vocos","latent"])
    parser.add_argument("--latent_frames",default=30,type=int)
    parser.add_argument("-to", "--tokenizer", default="pinyin", type=str, choices=["pinyin", "char"])

    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)
    parser.add_argument("-ch","--mel_channels",default=128,type=int)

    parser.add_argument("-t", "--testset", required=True)
    parser.add_argument("--latent_path",default=None)

    args = parser.parse_args()

    seed = args.seed
    dataset_name = args.dataset
    exp_name = args.expname
    ckpt_step = args.ckptstep
    ckpt_path = rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}.pt"
    mel_spec_type = args.mel_spec_type
    latent_frames = args.latent_frames if mel_spec_type == "latent" else 93.75
    tokenizer = args.tokenizer
    n_mel_channels = args.mel_channels
    latent_path = args.latent_path

    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling

    testset = args.testset

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = 2.0
    speed = 1.0
    use_truth_duration = False
    no_ref_audio = False

    if exp_name == "F5TTS_Base":
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

    if exp_name == "E2TTS_Base":
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    
    if "F5TTS_Small" in exp_name:
        model_cls = DiT
        model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)
    
    if exp_name == "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_latent_30hz_bzs102400_msk0.4-0.7":
        model_cls = DiT
        model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)
    
    if exp_name == "F5TTS_Small_vocos_char_LibriTTS_100_360_500_latent_30hz_bzs102400_msk0.4-0.7_convlayer0":
        model_cls = DiT
        model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=0)

    if testset == "ls_pc_test_clean":
        metalst = rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst"
        librispeech_test_clean_path = "/mnt/petrelfs/niuzhikang/data/LibriSpeech/test-clean/"  # test-clean path
        metainfo = get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path)

    elif testset == "seedtts_test_zh":
        metalst = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/seed-tts-eval/seedtts_testset/en/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    elif testset == "seedtts_test_en":
        metalst = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/seed-tts-eval/seedtts_testset/en/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    # path to save genereted wavs
    output_dir = (
        f"{rel_path}/"
        f"results/{exp_name}_{ckpt_step}/{testset}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )

    # -------------------------------------------------#

    use_ema = True
    if mel_spec_type in ["bigvgan", "vocos"]:
        prompts_all = get_inference_prompt(
            metainfo,
            speed=speed,
            tokenizer=tokenizer,
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
            mel_spec_type=mel_spec_type,
            target_rms=target_rms,
            use_truth_duration=use_truth_duration,
            infer_batch_size=infer_batch_size,
        )
    elif mel_spec_type == "latent":
        prompts_all = get_latent_inference_prompt(
            metainfo,
            speed=speed,
            tokenizer=tokenizer,
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=int(target_sample_rate//latent_frames), # not sure
            mel_spec_type=mel_spec_type,
            target_rms=target_rms,
            use_truth_duration=use_truth_duration,
            infer_batch_size=infer_batch_size,
            latent_frames=latent_frames,
            latent_path=latent_path
        )

    # Vocoder model
    local = False
    if mel_spec_type != "latent":
        if mel_spec_type == "vocos":
            vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
        elif mel_spec_type == "bigvgan":
            vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
        vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Model
    # n_mel_channels = 128 if mel_spec_type == "latent" else n_mel_channels
    print(f"mel_channels: {n_mel_channels}")
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels= n_mel_channels ,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # start batch inference
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = prompt
            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)

            # Inference
            with torch.inference_mode():
                generated, _ = model.sample(
                    cond=ref_mels, # cond是mel频谱图
                    text=final_text_list, # 文本
                    duration=total_mel_lens,
                    lens=ref_mel_lens,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    no_ref_audio=no_ref_audio,
                    seed=seed,
                )
                # Final result
                for i, gen in enumerate(generated):
                    gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
                    gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
                    if mel_spec_type in ["vocos","bigvgan"]:
                        if mel_spec_type == "vocos":
                            generated_wave = vocoder.decode(gen_mel_spec).cpu()
                        elif mel_spec_type == "bigvgan":
                            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
                        if ref_rms_list[i] < target_rms:
                            generated_wave = generated_wave * ref_rms_list[i] / target_rms
                            torchaudio.save(f"{output_dir}/{utts[i]}.wav", generated_wave, target_sample_rate)
                    elif mel_spec_type == "latent":
                        np.save(f"{output_dir}/{utts[i]}.npy",gen_mel_spec.cpu().numpy())
                        



    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60 :.2f} minutes.")


if __name__ == "__main__":
    main()
