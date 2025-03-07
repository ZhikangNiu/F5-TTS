export WANDB_API_KEY=16463fbb05d59374feba5b9e4b5dc705bfa4f7ba
export WANDB_MODE=offline
latent_frames=30
mel_spec_type=latent
batch_size_per_gpu=51200
conv_layers=4
learning_rate=7e-5
hydra_args="
++model.arch.conv_layers=${conv_layers}
++model.vocoder.is_local=True
++model.vocoder.local_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/vocos-mel-24khz
++datasets.name=LibriTTS_100_360_500
++datasets.batch_size_per_gpu=${batch_size_per_gpu}
++model.tokenizer=char
++model.mel_spec.mel_spec_type=${mel_spec_type}
++model.frac_lengths_mask=[0.7,1.0]
++model.mel_spec.n_mel_channels=128
++model.latent_frames=${latent_frames}
++hydra.run.dir=ckpts/F5TTS_Small_vocos_char_LibriTTS_100_360_500_${mel_spec_type}_${latent_frames}hz_bzs${batch_size_per_gpu}_msk0.7-1.0_convlayer${conv_layers}_lr${learning_rate}
++ckpts.save_dir=ckpts/F5TTS_Small_vocos_char_LibriTTS_100_360_500_${mel_spec_type}_${latent_frames}hz_bzs${batch_size_per_gpu}_msk0.7-1.0_convlayer${conv_layers}_lr${learning_rate}
++ckpts.exp_name=F5TTS_Small_vocos_char_LibriTTS_100_360_500_${mel_spec_type}_bzs${batch_size_per_gpu}_msk0.7-1.0_convlayer${conv_layers}_lr${learning_rate}
++optim.epochs=1000
++datasets.max_samples=64
++ckpts.save_per_updates=50000
++ckpts.last_per_steps=5000
++optim.learning_rate=${learning_rate}
"
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args
# accelerate launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args
# accelerate launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args

accelerate launch --config_file 4gpu.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args