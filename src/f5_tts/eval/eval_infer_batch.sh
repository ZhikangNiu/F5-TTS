set -e
source /mnt/petrelfs/niuzhikang/miniconda3/etc/profile.d/conda.sh
conda activate f5tts

# #!/bin/bash

# # e.g. F5-TTS, 16 NFE
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "seedtts_test_zh" -nfe 16
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "seedtts_test_en" -nfe 16
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "ls_pc_test_clean" -nfe 16

# # e.g. Vanilla E2 TTS, 32 NFE
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -t "seedtts_test_zh" -o "midpoint" -ss 0
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -t "seedtts_test_en" -o "midpoint" -ss 0
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -t "ls_pc_test_clean" -o "midpoint" -ss 0

# # etc.
# # accelerate 
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --main_process_port 23679  src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_latent_30hz" -t "ls_pc_test_clean" -nfe 16 -m "latent" -to char --ckptstep 30000 -d LibriTTS_100_360_500
# accelerate launch --main_process_port 23679  src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_latent_30hz_bzs102400" -t "ls_pc_test_clean" -nfe 16 -m "latent" -to char --ckptstep 200000 -d LibriTTS_100_360_500
# set -e
latent_frames=40
mel_channels=48
latent_path=/mnt/petrelfs/niuzhikang/descript-audio-codec/LibriSpeech/24khz_600x_8553_kl0.25_vae64_klwarmup_clamp_logvar/40hz_feat/test-clean/
latent_path=/mnt/petrelfs/niuzhikang/descript-audio-codec/LibriSpeech/24khz_600x_8553_kl1e-2_vae48_klwarmup_clamp_logvar/40hz_feat/test-clean/
for ckpt in 200000;do
    accelerate launch --main_process_port 23679  src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_latent_40hz_bzs102400_msk0.4-0.7_convlayer4_lr3e-4_kl1e-2_vae48_fix_rope" -t "ls_pc_test_clean" -nfe 32 -m "latent" -to char --ckptstep $ckpt -d LibriTTS_100_360_500 --latent_frames $latent_frames --mel_channels $mel_channels --latent_path $latent_path
    # accelerate launch --main_process_port 23679  src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_latent_30hz_bzs102400_msk0.4-0.7" -t "ls_pc_test_clean" -nfe 32 -m "latent" -to char --ckptstep $ckpt -d LibriTTS_100_360_500 --latent_frames 30
done
cd /mnt/petrelfs/niuzhikang/descript-audio-codec/
conda activate dac
for ckpt in 200000;do
    python scripts/recon_wave.py --path runs/2gpu/24khz_600x_8553_kl1e-2_vae48_klwarmup_clamp_logvar/ --input /mnt/petrelfs/niuzhikang/F5-TTS/results/F5TTS_Small_vocos_char_LibriTTS_100_360_500_latent_40hz_bzs102400_msk0.4-0.7_convlayer4_lr3e-4_kl1e-2_vae48_fix_rope_${ckpt} --model_tag 400k
    # accelerate launch --main_process_port 23679  src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_latent_30hz_bzs102400_msk0.4-0.7" -t "ls_pc_test_clean" -nfe 32 -m "latent" -to char --ckptstep $ckpt -d LibriTTS_100_360_500 --latent_frames 30
done