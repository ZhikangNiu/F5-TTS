hydra_args="
++model.vocoder.is_local=True
++model.vocoder.local_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/vocos-mel-24khz
++ckpts.log_samples=False
"
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_v1_vae_small $hydra_args
accelerate launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_v1_Base_vae $hydra_args
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_v1_Small $hydra_args