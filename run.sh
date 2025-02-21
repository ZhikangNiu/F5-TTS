set -e  
# export WANDB_API_KEY=16463fbb05d59374feba5b9e4b5dc705bfa4f7ba
# export WANDB_MODE=offline

dataset_name=LibriTTS_100_360_500
tokenizer=char
refine_type=conv
zero_init=True
silu_ff=True
text_padding_mask=True
mel_spec_type=vocos
batch_size_per_gpu=38400
exp_name=F5TTS_Small_${mel_spec_type}_${tokenizer}_${dataset_name}_${batch_size_per_gpu}_${refine_type}_text_mask_padding_${text_padding_mask}_zero_init_${zero_init}_silu_ff_${silu_ff}
wandb_project=CFM-F5TTS
vocoder_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/vocos-mel-24khz

hydra_args="
++model.vocoder.is_local=True
++model.vocoder.local_path=${vocoder_path}
++datasets.name=${dataset_name}
++model.tokenizer=${tokenizer}
++model.frac_lengths_mask=[0.7,1.0]
++model.arch.refine_type=${refine_type}
++model.arch.zero_init=${zero_init}
++model.arch.silu_ff=${silu_ff}
++model.arch.padding_mask=${text_padding_mask}
++hydra.run.dir=ckpts/${exp_name}
++ckpts.save_dir=ckpts/${exp_name}
++ckpts.exp_name=${exp_name}
++ckpts.wandb_project=${wandb_project}
++optim.epochs=686
"
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args

accelerate launch --config_file 4gpu.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args
