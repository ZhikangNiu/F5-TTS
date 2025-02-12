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
set -e
for ckpt in 1100000 1000000 600000 700000 800000 900000;do
    accelerate launch --main_process_port 23679  src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_latent_30hz_bzs51200_msk0.7-1.0_convlayer4_lr7e-5" -t "ls_pc_test_clean" -nfe 32 -m "latent" -to char --ckptstep $ckpt -d LibriTTS_100_360_500 --latent_frames 30
    # accelerate launch --main_process_port 23679  src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_latent_30hz_bzs102400_msk0.4-0.7" -t "ls_pc_test_clean" -nfe 32 -m "latent" -to char --ckptstep $ckpt -d LibriTTS_100_360_500 --latent_frames 30
done
