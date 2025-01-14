import os
import pytest
import shutil
from utils.experiment_pipeline import ExperimentPipeline
from data.particle_clouds.jets import JetDataModule
from utils.helpers import SimpleLogger as log
log.warnings_off()


RES_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
OUTPUT_PATH = "/home/df630/Multimodal-Bridges/tests/output"


def test_new_experiment():
    CONFIG_PATH = os.path.join(RES_PATH, "config.yaml")
    new_exp = ExperimentPipeline(JetDataModule, config=CONFIG_PATH)
    new_exp.train()
    # shutil.rmtree(new_exp.trainer.config.path)


def test_resume_from_checkpoint():
    EXP_PATH = f"{OUTPUT_PATH}/multimodal-jets/0bfa9c6b34ca404aa91a3b0cdb5cff6e"
    resume_exp = ExperimentPipeline(
        JetDataModule,
        config={"trainer": {"max_epochs": 50}},
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
    )
    resume_exp.train()


def test_experiment_generation():
    EXP_PATH = f"{OUTPUT_PATH}/multimodal-jets/0bfa9c6b34ca404aa91a3b0cdb5cff6e"
    pipeline = ExperimentPipeline(
        JetDataModule,
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
        devices=[0],
    )
    pipeline.generate()


if __name__ == "__main__":
    test_experiment_generation()
