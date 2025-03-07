# export WANDB_API_KEY=16463fbb05d59374feba5b9e4b5dc705bfa4f7ba
# export WANDB_MODE=offline

hydra_args="
++model.vocoder.is_local=True
++model.vocoder.local_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/vocos-mel-24khz
++datasets.name=Emilia_ZH_EN
++model.tokenizer=pinyin
++model.frac_lengths_mask=[0.7,1.0]
++model.arch.refine_type=conv
++model.arch.zero_init=True
++model.arch.silu_ff=True
++model.arch.qk_norm=False
++hydra.run.dir=ckpts/F5TTS_Small_vocos_Emilia_ZH_EN_pinyin_conv_mask_padding_fix_rope_only_zero_init_adaln_silu_ff_qk_norm
++ckpts.save_dir=ckpts/F5TTS_Small_vocos_Emilia_ZH_EN_pinyin_conv_mask_padding_fix_rope_only_zero_init_adaln_silu_ff_qk_norm
++ckpts.exp_name=F5TTS_Small_vocos_Emilia_ZH_EN_pinyin_conv_mask_padding_fix_rope_only_zero_init_adaln_silu_ff_qk_norm
++optim.epochs=12
"
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args
# accelerate launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args

accelerate launch --config_file 4gpu.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args
