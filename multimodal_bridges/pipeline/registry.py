import torch
from datamodules.aoj import AspenOpenJets
from distributions.noise import Noise, SourceUniform, SourceDegenerate, SourceMasked, SourceDataDriven
from model.bridges import UniformFlow, SchrodingerBridge, TelegraphBridge
from encoders.multimodal_epic import UniModalEPiC, MultiModalEPiC, MultiModalFusedEPiC

registered_datasets = {
    "AspenOpenJets": AspenOpenJets,
    }

registered_distributions = {
    "Noise": Noise,
    "SourceUniform": SourceUniform,
    "SourceDegenerate": SourceDegenerate,
    "SourceMasked": SourceMasked,
    "SourceDataDriven": SourceDataDriven,
}

registered_bridges = {
    "UniformFlow": UniformFlow,
    "SchrodingerBridge": SchrodingerBridge,
    "TelegraphBridge": TelegraphBridge,
}

registered_models = {
    "UniModalEPiC": UniModalEPiC,
    "MultiModalEPiC": MultiModalEPiC,
    "MultiModalFusedEPiC": MultiModalFusedEPiC,
}

registered_optimizers = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
}

registered_schedulers = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
}