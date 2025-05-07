# export WANDB_API_KEY=16463fbb05d59374feba5b9e4b5dc705bfa4f7ba
# export WANDB_MODE=offline
hydra_args="
++model.vocoder.is_local=True
++model.vocoder.local_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/vocos-mel-24khz
"
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train_e2e.py -cn e2e_F5TTS_v1_Small_ldm64_40hz_lr1e-4_bsz51200 $hydra_args
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_v1_Small_vocos_char_LibriTTS_100_360_500 $hydra_args
accelerate launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_v1_Small_vocos_char_LibriTTS_100_360_500 $hydra_args
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_v1_Small $hydra_args