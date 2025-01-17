import os
import pytest
import shutil
from utils.experiment_pipeline import ExperimentPipeline
from data.particle_clouds.jets import JetDataModule
from utils.helpers import SimpleLogger as log

log.warnings_off()


RES_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
OUTPUT_PATH = "/home/df630/Multimodal-Bridges/tests/output"


def test_new_experiment_continuous():
    CONFIG_PATH = os.path.join(RES_PATH, "config_continuous.yaml")
    new_exp = ExperimentPipeline(JetDataModule, config=CONFIG_PATH)
    new_exp.train()


def test_resume_from_checkpoint_continuous():
    for subdir in os.listdir(f"{OUTPUT_PATH}/multimodal-jets"):
        EXP_PATH = f"{OUTPUT_PATH}/multimodal-jets/{subdir}"

    resume_exp = ExperimentPipeline(
        JetDataModule,
        config={"trainer": {"max_epochs": 50}},
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
    )
    resume_exp.train()


def test_experiment_generation_continuous_single_gpu():
    for subdir in os.listdir(f"{OUTPUT_PATH}/multimodal-jets"):
        EXP_PATH = f"{OUTPUT_PATH}/multimodal-jets/{subdir}"

    pipeline = ExperimentPipeline(
        JetDataModule,
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
        devices=[0],
    )
    pipeline.generate()

    for subdir in os.listdir(f"{OUTPUT_PATH}/multimodal-jets"):
        EXP_PATH = f"{OUTPUT_PATH}/multimodal-jets/{subdir}"
        shutil.rmtree(EXP_PATH)

def test_experiment_generation_continuous_multi_gpu():
    for subdir in os.listdir(f"{OUTPUT_PATH}/multimodal-jets"):
        EXP_PATH = f"{OUTPUT_PATH}/multimodal-jets/{subdir}"

    pipeline = ExperimentPipeline(
        JetDataModule,
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
        devices=[0,3],
    )
    pipeline.generate()

    for subdir in os.listdir(f"{OUTPUT_PATH}/multimodal-jets"):
        EXP_PATH = f"{OUTPUT_PATH}/multimodal-jets/{subdir}"
        shutil.rmtree(EXP_PATH)

if __name__ == "__main__":
    test_experiment_generation_continuous_multi_gpu()
