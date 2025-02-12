set -e
pip install --upgrade faster-whisper==1.1.1 && pip install --upgrade ctranslate2==4.5.0
if [ ! -d "$HOME/.cache/torch" ]; then
    mkdir -p "$HOME/.cache/torch"
fi
cp -r /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/hub ~/.cache/torch
cp -r /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/s3prl ~/.cache
gpu_nums=4 # default 4
nfe_step=32
mel_spec_type=latent
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python3 -c 'import os; import torch; print(os.path.dirname(torch.__file__) +"/lib")'`
# exp_name=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/F5TTS_Small_vocos_char_LibriTTS_100_360_500_latent_30hz_bzs102400_msk0.4-0.7_convlayer0_
exp_name=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/F5TTS_Small_vocos_char_LibriTTS_100_360_500_latent_30hz_bzs51200_msk0.7-1.0_convlayer4_lr7e-5_
librispeech_test_clean_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriSpeech/test-clean
infer_setting=seed0_euler_nfe${nfe_step}_${mel_spec_type}_ss-1_cfg2.0_speed1.0

log_dir="$exp_name/logs"
mkdir -p "$log_dir"


# for weight in 0.75 1.25 ; do
# for ckpt in 100000 200000 300000; do
for ckpt in 1100000 1000000;do
    for task in wer;do
        log_file="${log_dir}/${ckpt}_log_vocos.txt"
        echo "Task: $task" | tee -a "$log_file"
        gen_wav_dir="$exp_name${ckpt}/ls_pc_test_clean/${infer_setting}/"
        # gen_wav_dir="
        echo $gen_wav_dir | tee -a "$log_file"
        python src/f5_tts/eval/eval_librispeech_test_clean.py --eval_task $task -g $gen_wav_dir -p $librispeech_test_clean_path -n $gpu_nums --local | tee -a "$log_file"
    done
done