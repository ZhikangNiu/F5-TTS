#!/bin/bash
set -e
# e.g. F5-TTS, 16 NFE
# pyd `which accelerate` launch --config_file debug.yaml src/f5_tts/eval/eval_infer_batch.py -s 1 -n "F5TTS_Base" -t "ls_pc_test_clean" -c 1200000 -d Emilia_ZH_EN -to pinyin --nfestep 16
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "seedtts_test_en" -nfe 16
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "ls_pc_test_clean" -nfe 16

# # e.g. Vanilla E2 TTS, 32 NFE
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -t "seedtts_test_zh" -o "midpoint" -ss 0
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -t "seedtts_test_en" -o "midpoint" -ss 0
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -t "ls_pc_test_clean" -o "midpoint" -ss 0
# exp_name=F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_mask0.7-1.0_baseline_4gpu_zero_start
# accelerate_config=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/4gpu.yaml
accelerate_config=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/1node2gpu.yaml
# accelerate launch --config_file "$accelerate_config" src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t ls_pc_test_clean -nfe 32 -c 1200000 -d Emilia_ZH_EN -to pinyin --nfestep 8
start=0
end=5
step=0.25

value=$start
while (( $(echo "$value <= $end" | bc -l) )); do
    for data in ls_pc_test_clean; do
        accelerate launch --config_file "$accelerate_config" --main_process_port 29061 src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "$data" -nfe 32 -c 1200000 -d Emilia_ZH_EN -to pinyin --cfg $value
    done
    value=$(echo "$value + $step" | bc)
done


# value=$start
# while (( $(echo "$value <= $end" | bc -l) )); do
#     for data in seedtts_test_zh seedtts_test_en ls_pc_test_clean; do
#         accelerate launch --config_file "$accelerate_config" src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "$data" -nfe 32 -c 1200000 -d Emilia_ZH_EN -to pinyin --cfg $value
#     done
#     value=$(echo "$value + $step" | bc)
# done

# for data in seedtts_test_zh seedtts_test_en ls_pc_test_clean; do
# for data in ls_pc_test_clean; do
#     for ckpt in 300000; do
#         accelerate launch --config_file "$accelerate_config" --main_process_port=29501 src/f5_tts/eval/eval_infer_batch.py -s 0 -n "$exp_name" -t "$data" -nfe 32 -c "$ckpt" -d LibriTTS_100_360_500 -to char
#     done
# done

