#!/bin/bash
set -e
# accelerate_config=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/4gpu.yaml
data=ls_pc_test_clean
accelerate_config=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/f5-prefix/debug.yaml
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t ls_pc_test_clean -nfe 32 -c 1200000 -d Emilia_ZH_EN -to pinyin --nfestep 8
# for ckpt in 400000;do
#     accelerate launch --config_file "$accelerate_config" --main_process_port 29061 src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_conv_mask_padding_rmsnorm" -t "$data" -nfe 32 -c $ckpt -d LibriTTS_100_360_500 -to char
# done
# for exp_seed in 0 1 2;do
for ckpt in 700000;do
    for cfg in 1.0 1.5 2.5;do
        accelerate launch --config_file "$accelerate_config" --main_process_port 29061 src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_v1_Small_vocos_char_LibriTTS_100_360_500" -t "$data" -nfe 32 -c $ckpt --cfg $cfg
    done
done
# e.g. F5-TTS, 16 NFE
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_v1_Base" -t "seedtts_test_zh" -nfe 16
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_v1_Base" -t "seedtts_test_en" -nfe 16
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_v1_Base" -t "ls_pc_test_clean" -nfe 16

# # e.g. Vanilla E2 TTS, 32 NFE
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -c 1200000 -t "seedtts_test_zh" -o "midpoint" -ss 0
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -c 1200000 -t "seedtts_test_en" -o "midpoint" -ss 0
# accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -c 1200000 -t "ls_pc_test_clean" -o "midpoint" -ss 0

# # e.g. evaluate F5-TTS 16 NFE result on Seed-TTS test-zh
# python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l zh --gen_wav_dir results/F5TTS_v1_Base_1250000/seedtts_test_zh/seed0_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0 --gpu_nums 8
# python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l zh --gen_wav_dir results/F5TTS_v1_Base_1250000/seedtts_test_zh/seed0_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0 --gpu_nums 8
# python src/f5_tts/eval/eval_utmos.py --audio_dir results/F5TTS_v1_Base_1250000/seedtts_test_zh/seed0_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0

# etc.
