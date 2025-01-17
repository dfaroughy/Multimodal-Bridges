import os
import pytest
import shutil
import h5py
from utils.experiment_pipeline import ExperimentPipeline
from data.particle_clouds.jets import JetDataModule
from utils.helpers import SimpleLogger as log

log.warnings_off()

@pytest.fixture
def modality():
    return 'multi-modal'

@pytest.fixture
def devices():
    return [0,3]

def test_experiment_multimodal(modality, devices):
   
    OUTPUT_PATH = "/home/df630/Multimodal-Bridges/tests/output/multimodal-jets"
    CONFIG_PATH = f"/home/df630/Multimodal-Bridges/tests/resources/config_{modality}.yaml"
    
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH) 

    # 1. Create a new experiment:
    new_exp = ExperimentPipeline(JetDataModule, config=CONFIG_PATH, devices=devices)
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
        config={"trainer": {"max_epochs": 50}, "checkpoints": {"filename": "better"}},
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
        devices=devices,
    )
    resume_exp.train()

    assert os.path.exists(f"{EXP_PATH}/checkpoints/better.ckpt")

    # 3. Generate data from resumed experiment:
    pipeline = ExperimentPipeline(
        JetDataModule,
        config={"data": {"batch_size": 20, "num_jets": 60}},
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
        devices=devices,
    )
    pipeline.generate()

    assert os.path.exists(f"{EXP_PATH}/data/generated_sample.h5")
    assert os.path.exists(f"{EXP_PATH}/data/test_sample.h5")
    assert os.path.exists(f"{EXP_PATH}/metrics/performance_metrics.json")

    with h5py.File(f"{EXP_PATH}/data/generated_sample.h5", "r") as f:

        if modality in ["continuous", "multi-modal"]:
            continuous = f["continuous"]
            assert continuous.shape == (60, 128, 3)

        if modality in ["discrete", "multi-modal"]:
            discrete = f["discrete"]
            assert discrete.shape == (60, 128, 6)


if __name__ == "__main__":
    test_experiment_multimodal()
