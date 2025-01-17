import torch
from model.bridges import LinearUniformBridge, SchrodingerBridge, TelegraphBridge
from encoders.epic import MultiModalEPiC
from encoders.particle_transformer import MultiModalParticleTransformer

registered_bridges = {
    "LinearUniformBridge": LinearUniformBridge,
    "SchrodingerBridge": SchrodingerBridge,
    "TelegraphBridge": TelegraphBridge,
}

registered_models = {
    "MultiModalEPiC": MultiModalEPiC,
    "MultiModalParticleTransformer": MultiModalParticleTransformer,
}

registered_optimizers = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
}

registered_schedulers = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
}