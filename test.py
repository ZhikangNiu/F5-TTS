import hydra

model_cls = hydra.utils.get_class("f5_tts.model.DiTMoE")
model_cls = hydra.utils.get_class("f5_tts.model.DiT")
# print(model_cls())