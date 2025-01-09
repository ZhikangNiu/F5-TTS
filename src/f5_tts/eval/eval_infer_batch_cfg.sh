#!/bin/bash
set -e
apt-get install bc
# exp_name=F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_mask0.7-1.0_baseline_4gpu_zero_start
exp_name=F5TTS_Base
ckpt=1200000
# accelerate_config=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/4gpu.yaml
accelerate_config=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/1node2gpu.yaml
start=0
end=5
step=0.25

value=$start
while (( $(echo "$value <= $end" | bc -l) )); do
    for data in seedtts_test_zh seedtts_test_en ls_pc_test_clean; do
        accelerate launch --config_file "$accelerate_config" --main_process_port 29061 src/f5_tts/eval/eval_infer_batch.py -s 0 -n "$exp_name" -t "$data" -nfe 32 -c $ckpt -d Emilia_ZH_EN -to pinyin --cfg $value
    done
    value=$(echo "$value + $step" | bc)
done


# # cfg exp

# 设置路径
results_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/${exp_name}_${ckpt}/ls_pc_test_clean
librispeech_test_clean_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriSpeech/test-clean

# 配置起始和结束值
start=0
end=5
step=0.25
gpu_nums=4

# 创建日志文件夹
log_dir="$exp_name/logs"
mkdir -p "$log_dir"

# 配置初始值
value=$start

# 开始循环
while (( $(echo "$value <= $end" | bc -l) )); do
    # 判断整数和小数
    if (( $(echo "$value == ${value%.*}" | bc -l) )); then
        formatted_value=$(printf "%.1f" "$value") # 整数格式化为 1.0
    else
        formatted_value=$(printf "%.2f" "$value" | sed 's/0*$//') # 去掉小数末尾多余的0
    fi

    # 设置路径和日志文件
    gen_wav_dir="$exp_name/seed0_euler_nfe32_vocos_ss-1_cfg${formatted_value}_speed1.0"
        # gen_wav_dir="$exp_name/seed0_euler_nfe16_vocos_ss-1_cfg${formatted_value}_speed1.0"
    log_file="$log_dir/cfg_${formatted_value}_log.txt"
    
    # 打印并执行任务
    echo "Processing: $gen_wav_dir" | tee -a "$log_file"
    for task in wer sim; do
        echo "Task: $task" | tee -a "$log_file"
        python src/f5_tts/eval/eval_librispeech_test_clean.py \
            --eval_task $task \
            -g $gen_wav_dir \
            -p $librispeech_test_clean_path \
            -n $gpu_nums --local | tee -a "$log_file"
    done

    # 更新 value
    value=$(echo "$value + $step" | bc)
done
