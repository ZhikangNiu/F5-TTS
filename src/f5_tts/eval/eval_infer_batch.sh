#!/bin/bash
set -e
accelerate_config=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/4gpu.yaml
data=ls_pc_test_clean
# accelerate_config=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/1node2gpu.yaml
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t ls_pc_test_clean -nfe 32 -c 1200000 -d Emilia_ZH_EN -to pinyin --nfestep 8
# for ckpt in 400000;do
#     accelerate launch --config_file "$accelerate_config" --main_process_port 29061 src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_conv_mask_padding_rmsnorm" -t "$data" -nfe 32 -c $ckpt -d LibriTTS_100_360_500 -to char
# done
# for exp_seed in 0 1 2;do
for ckpt in 400000;do
    accelerate launch --config_file "$accelerate_config" --main_process_port 29061 src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_conv_mask_padding_fix_rope_only_zero_init_adaln_silu_ff_qk_norm" -t "$data" -nfe 32 -c $ckpt -d LibriTTS_100_360_500 -to char
done
# done
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` --config_file debug.yaml src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_conv_mask_padding" -t "$data" -nfe 32 -c $ckpt -d LibriTTS_100_360_500 -to char
