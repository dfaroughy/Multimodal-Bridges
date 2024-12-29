import torch
from bridges import LinearUniformBridge, SchrodingerBridge, TelegraphBridge
from models.epic import MultiModalEPiC, EPiC

registered_bridges = {
    "LinearUniformBridge": LinearUniformBridge,
    "SchrodingerBridge": SchrodingerBridge,
    "TelegraphBridge": TelegraphBridge,
}

registered_models = {
    "MultiModalEPiC": MultiModalEPiC,
    "EPiC": EPiC,
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