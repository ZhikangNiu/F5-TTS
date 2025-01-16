#!/bin/bash
set -e
accelerate_config=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/4gpu.yaml
data=ls_pc_test_clean
# accelerate_config=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/1node2gpu.yaml
# accelerate launch --config_file "$accelerate_config" src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t ls_pc_test_clean -nfe 32 -c 1200000 -d Emilia_ZH_EN -to pinyin --nfestep 8
for ckpt in 100000 200000;do
    accelerate launch --config_file "$accelerate_config" --main_process_port 29061 src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_conv_bilstm_skip_false" -t "$data" -nfe 32 -c $ckpt -d LibriTTS_100_360_500 -to char
done

