export WANDB_API_KEY=16463fbb05d59374feba5b9e4b5dc705bfa4f7ba
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cfg_files=$1
acc_cfg=8gpu.yaml
hydra_args="
++model.vocoder.is_local=True
++model.vocoder.local_path=/inspire/hdd/global_user/chenxie-25019/download_ckpts/vocos-mel-24khz 
"
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_v1_vae_small $hydra_args
# accelerate launch --config_file 4gpu.yaml src/f5_tts/train/train.py -cn F5TTS_v1_Small_vocos_upsample_char_LibriTTS_100_360_500 $hydra_args
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file 1gpu.yaml src/f5_tts/train/train.py -cn F5TTS_v1_Base $hydra_args
accelerate launch --config_file $acc_cfg src/f5_tts/train/train.py -cn $cfg_files $hydra_args
