import os
import pytest
import shutil
import h5py

from pipeline.helpers import SimpleLogger as log
from pipeline.experiment import ExperimentPipeline
from datamodules.particle_clouds.jetmodule import JetDataModule

log.warnings_off()


@pytest.fixture
def modality():
    return "multi-modal"


@pytest.fixture
def devices():
    return [0, 3]


def test_new_experiment_multimodal():
    modality, devices = "multi-modal", [0, 1]

    OUTPUT_PATH = "/home/df630/Multimodal-Bridges/tests/output/multimodal-jets"
    CONFIG_PATH = (
        f"/home/df630/Multimodal-Bridges/tests/resources/config_{modality}.yaml"
    )

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    # 1. Create a new experiment:
    new_exp = ExperimentPipeline(
        JetDataModule,
        config=CONFIG_PATH,
        strategy="ddp_find_unused_parameters_true",
        devices=devices,
        tags="pytest",
    )
    new_exp.train()


def test_experiment_multimodal():
    modality, devices = "multi-modal", 1

    OUTPUT_PATH = "/home/df630/Multimodal-Bridges/tests/output/multimodal-jets"
    CONFIG_PATH = (
        f"/home/df630/Multimodal-Bridges/tests/resources/config_{modality}.yaml"
    )

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    # 1. Create a new experiment:
    new_exp = ExperimentPipeline(
        JetDataModule,
        config=CONFIG_PATH,
        devices=devices,
        strategy="ddp_find_unused_parameters_true",
        tags="pytest",
    )
    new_exp.train()

    for subdir in os.listdir(f"{OUTPUT_PATH}"):
        EXP_PATH = f"{OUTPUT_PATH}/{subdir}"

    assert os.path.exists(f"{EXP_PATH}/config.yaml")
    assert os.path.exists(f"{EXP_PATH}/metadata.json")
    assert os.path.exists(f"{EXP_PATH}/checkpoints/last.ckpt")
    assert os.path.exists(f"{EXP_PATH}/checkpoints/best.ckpt")

    # 2. Resume from checkpoint:
    resume_exp = ExperimentPipeline(
        JetDataModule,
        config={"trainer": {"max_epochs": 10}, "checkpoints": {"filename": "better"}},
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
        strategy="ddp_find_unused_parameters_true",
        devices=devices,
    )
    resume_exp.train()

    # 3. Generate data from resumed experiment:
    pipeline = ExperimentPipeline(
        JetDataModule,
        config={"data": {"batch_size": 5, "num_jets": 20}},
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
        devices=devices,
        strategy="ddp_find_unused_parameters_true",
    )
    pipeline.generate()

    # assert os.path.exists(f"{EXP_PATH}/data/generated_sample.h5")
    # assert os.path.exists(f"{EXP_PATH}/data/test_sample.h5")
    # assert os.path.exists(f"{EXP_PATH}/metrics/performance_metrics.json")

    with h5py.File(f"{EXP_PATH}/data/generated_sample.h5", "r") as f:
        if modality in ["continuous", "multi-modal"]:
            continuous = f["continuous"]
            assert continuous.shape == (20, 128, 3)

        if modality in ["discrete", "multi-modal"]:
            discrete = f["discrete"]
            assert discrete.shape == (20, 128, 1)


if __name__ == "__main__":
    test_experiment_multimodal()
