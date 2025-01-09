set -e
pip install --upgrade faster-whisper && pip install --upgrade ctranslate2
gpu_nums=2 # default 4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python3 -c 'import os; import torch; print(os.path.dirname(torch.__file__) +"/lib")'`
exp_name=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/F5TTS_Base_1200000/ls_pc_test_clean
librispeech_test_clean_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriSpeech/test-clean
infer_setting=seed0_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0

log_dir="$exp_name/logs"
mkdir -p "$log_dir"

# # for ckpt in 400000 500000; do
for weight in 0.75 1.25 ; do
    for task in wer sim;do
        log_file="${log_dir}/${ckpt}_log.txt"
        echo "Task: $task" | tee -a "$log_file"
        gen_wav_dir="$exp_name/${infer_setting}_audio${weight}/"
    
        echo $gen_wav_dir | tee -a "$log_file"
        python src/f5_tts/eval/eval_librispeech_test_clean.py --eval_task $task -g $gen_wav_dir -p $librispeech_test_clean_path -n 2 --local | tee -a "$log_file"
    done
done


