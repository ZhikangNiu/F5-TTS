export WANDB_API_KEY=16463fbb05d59374feba5b9e4b5dc705bfa4f7ba
# export WANDB_MODE=offline
latent_frames=40
vae_dim=48 # 128
kl_weight=1e-2
# latent_path=/mnt/petrelfs/niuzhikang/descript-audio-codec/LibriTTS/24khz_800x_8544_kl5e-5_vae32_clamp_logvar/30hz_feat/
# latent_path=/mnt/petrelfs/niuzhikang/descript-audio-codec/LibriTTS/24khz_600x_8553_kl0.25_vae64_klwarmup_clamp_logvar/40hz_feat/
latent_path=/mnt/petrelfs/niuzhikang/descript-audio-codec/LibriTTS/24khz_600x_8553_kl1e-2_vae48_klwarmup_clamp_logvar/40hz_feat/
# latent_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/LibriTTS/24khz_800x_8544_kl1e-2_vae128_clamp_logvar/30hz_feat
mel_spec_type=latent
batch_size_per_gpu=102400
conv_layers=4
learning_rate=3e-4
hydra_args="
++model.arch.conv_layers=${conv_layers}
++datasets.name=LibriTTS_100_360_500
++datasets.batch_size_per_gpu=${batch_size_per_gpu}
++model.tokenizer=char
++model.mel_spec.mel_spec_type=${mel_spec_type}
++model.frac_lengths_mask=[0.4,0.7]
++model.mel_spec.n_mel_channels=${vae_dim}
++model.latent_frames=${latent_frames}
++model.latent_path=${latent_path}
++hydra.run.dir=ckpts/F5TTS_Small_vocos_char_LibriTTS_100_360_500_${mel_spec_type}_${latent_frames}hz_bzs${batch_size_per_gpu}_msk0.4-0.7_convlayer${conv_layers}_lr${learning_rate}_kl${kl_weight}_vae${vae_dim}_fix_rope
++ckpts.save_dir=ckpts/F5TTS_Small_vocos_char_LibriTTS_100_360_500_${mel_spec_type}_${latent_frames}hz_bzs${batch_size_per_gpu}_msk0.4-0.7_convlayer${conv_layers}_lr${learning_rate}_kl${kl_weight}_vae${vae_dim}_fix_rope
++ckpts.exp_name=F5TTS_Small_vocos_char_LibriTTS_100_360_500_${mel_spec_type}_bzs${batch_size_per_gpu}_msk0.4-0.7_convlayer${conv_layers}_lr${learning_rate}_kl${kl_weight}_vae${vae_dim}_fix_rope
++optim.epochs=1000
++datasets.max_samples=256
++ckpts.save_per_updates=50000
++ckpts.last_per_steps=5000
++optim.learning_rate=${learning_rate}
"
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args
# accelerate launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args
# accelerate launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args

accelerate launch --config_file 4gpu.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args