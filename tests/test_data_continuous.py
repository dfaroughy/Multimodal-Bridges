import os
import pytest
import torch
import torch.nn.functional as F

from utils.helpers import get_from_json
from utils.helpers import SimpleLogger as log

log.warnings_off()

from utils.configs import ExperimentConfigs
from data.particle_clouds.jets import JetDataModule

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
        assert batch.source.continuous.shape[0] == config.data.batch_size
        assert batch.source.continuous.shape[1] == config.data.max_num_particles
        assert batch.source.continuous.shape[2] == config.data.dim_continuous
        assert batch.source.discrete is None
        assert batch.source.mask.shape[2] == 1

        assert batch.target.continuous.shape[0] == config.data.batch_size
        assert batch.target.continuous.shape[1] == config.data.max_num_particles
        assert batch.target.continuous.shape[2] == config.data.dim_continuous
        assert batch.target.discrete is None
        assert batch.target.mask.shape[2] == 1

    jets.config.data.num_jets = 90
    jets.setup(stage="predict")
    predict_dataloader = jets.predict_dataloader()

    assert (
        len(predict_dataloader) == jets.config.data.num_jets // config.data.batch_size
    )

    for batch in predict_dataloader:
        assert batch.source.continuous.shape[0] == config.data.batch_size
        assert batch.source.continuous.shape[1] == config.data.max_num_particles
        assert batch.source.continuous.shape[2] == config.data.dim_continuous
        assert batch.source.discrete is None
        assert batch.source.mask.shape[2] == 1
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
    assert jets.source.continuous.shape == torch.Size([N, M, D0])
    assert jets.source.mask.shape == torch.Size([N, M, 1])
    assert jets.target.continuous.shape == torch.Size([N, M, D0])
    assert jets.target.mask.shape == torch.Size([N, M, 1])


def test_data_processing_closure():
    config = ExperimentConfigs(CONFIG_PATH)
    jets = JetDataModule(config=config, preprocess=False)
    jets_preprocessed = JetDataModule(config=config, metadata_path=OUTPUT_PATH)

    jets.setup()
    jets_preprocessed.setup()
    prep_continuous = config.data.target_preprocess_continuous
    stats = get_from_json("target_data_stats", OUTPUT_PATH, "metadata.json")

    jets_preprocessed.target.postprocess(prep_continuous, None, **stats)

    assert torch.allclose(jets.target.continuous, jets_preprocessed.target.continuous)


if __name__ == "__main__":
    test_data_processing_closure()
