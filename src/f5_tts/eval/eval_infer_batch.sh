set -e

latent_frames=40
mel_channels=64
latent_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/LibriSpeech/test-clean/24khz_600x_8553_kl1e-3_vae64_klwarmup_6khrs_no_pre_post/40hz_feat/
for ckpt in 300000 200000 100000;do
    accelerate launch --main_process_port 23679  src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_v1_Small_vocos_char_LibriTTS_100_360_500_latent_40hz_bzs102400_msk0.4-0.7_convlayer4_lr3e-4_kl1e-3_vae64_rope" -t "ls_pc_test_clean" -nfe 32 -m "latent" -to char --ckptstep $ckpt -d LibriTTS_100_360_500 --latent_frames $latent_frames --mel_channels $mel_channels --latent_path $latent_path
    # accelerate launch --main_process_port 23679  src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_latent_30hz_bzs102400_msk0.4-0.7" -t "ls_pc_test_clean" -nfe 32 -m "latent" -to char --ckptstep $ckpt -d LibriTTS_100_360_500 --latent_frames 30
done
