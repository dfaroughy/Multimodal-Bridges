import os
import pytest
import torch

from pipeline.helpers import get_from_json
from pipeline.helpers import SimpleLogger as log
from pipeline.configs import ExperimentConfigs
from datamodules.particle_clouds.jetmodule import JetDataModule

log.warnings_off()

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
OUTPUT_PATH = "/home/df630/Multimodal-Bridges/tests/output"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config_continuous.yaml")


def test_configs():
    config = ExperimentConfigs(CONFIG_PATH)
    assert config.data.dim_discrete == 0
    assert config.data.vocab_size == 0
    assert config.data.dim_context_discrete == 0
    assert config.data.vocab_size_context == 0
    assert config.encoder.dim_emb_discrete == 0
    assert config.encoder.dim_emb_context_discrete == 0
    assert config.encoder.embed_type_discrete is None
    assert config.encoder.embed_type_context_discrete is None
    assert config.model.bridge_discrete is None


def test_databatch():
    config = ExperimentConfigs(CONFIG_PATH)
    jets = JetDataModule(config=config)

    jets.setup(stage="fit")
    train_dataloader = jets.train_dataloader()

    assert (
        len(train_dataloader)
        == jets.config.data.num_jets
        * config.data.split_ratios[0]
        // config.data.batch_size
    )

    for batch in train_dataloader:
        assert batch.source.continuous == None
        assert batch.source.continuous == None
        assert batch.source.continuous == None
        assert batch.source.discrete == None
        assert batch.source.mask == None

        assert batch.target.continuous.shape[0] == config.data.batch_size
        assert batch.target.continuous.shape[1] == config.data.max_num_particles
        assert batch.target.continuous.shape[2] == config.data.dim_continuous
        assert batch.target.discrete is None
        assert batch.target.mask.shape[2] == 1

    jets.setup(stage="predict")
    predict_dataloader = jets.predict_dataloader()

    assert len(predict_dataloader) == jets.config.data.num_jets // config.data.batch_size


    for batch in predict_dataloader:
        assert batch.target.continuous.shape[0] == config.data.batch_size
        assert batch.target.continuous.shape[1] == config.data.max_num_particles
        assert batch.target.continuous.shape[2] == config.data.dim_continuous
        assert batch.target.discrete is None
        assert batch.target.mask.shape[2] == 1


def test_data_shapes():
    config = ExperimentConfigs(CONFIG_PATH)
    jets = JetDataModule(config=config)
    jets.setup()
    N = config.data.num_jets
    M = config.data.max_num_particles
    D0 = config.data.dim_continuous

    assert jets.target.continuous.shape == torch.Size([N, M, D0])
    assert jets.target.mask.shape == torch.Size([N, M, 1])


if __name__ == "__main__":
    test_databatch()
