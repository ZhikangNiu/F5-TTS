export WANDB_API_KEY=16463fbb05d59374feba5b9e4b5dc705bfa4f7ba
export WANDB_MODE=offline

hydra_args="
++model.vocoder.is_local=True
++model.vocoder.local_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/vocos-mel-24khz
++datasets.name=LibriTTS_100_360_500
++model.tokenizer=char
++model.frac_lengths_mask=[0.7,1.0]
++model.arch.refine_type=transformer_encoder
++hydra.run.dir=ckpts/F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_text_encoder_transformer_head8_padding_mask
++ckpts.save_dir=ckpts/F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_text_encoder_transformer_head8_padding_mask
++ckpts.exp_name=F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_text_encoder_transformer_head8_padding_mask
++optim.epochs=686
"
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args
# accelerate launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args

accelerate launch --config_file 4gpu.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args
