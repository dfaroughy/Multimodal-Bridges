import os
import pytest
from pathlib import Path
from utils.experiment_pipeline import ExperimentPipeline
import shutil

RES_PATH = "/home/df630/Multimodal-Bridges/tests/resources"


def test_new_experiment():
    CONFIG_PATH = os.path.join(RES_PATH, "config.yaml")
    new_exp = ExperimentPipeline(config=CONFIG_PATH)
    new_exp.train()
    assert Path(new_exp.trainer.config.path).exists()
    shutil.rmtree(new_exp.trainer.config.path)


def test_resume_from_checkpoint():
    EXP_PATH = f"{RES_PATH}/output/multimodal-jets/b9430ef34f0e48da987ea80e495b4263"
    resume_exp = ExperimentPipeline(
        config={"trainer": {"max_epochs": 30}},
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
    )
    resume_exp.train()


def test_experiment_generation():
    EXP_PATH = f"{RES_PATH}/output/multimodal-jets/b9430ef34f0e48da987ea80e495b4263"
    CONFIG_PATH = f"{RES_PATH}/config_gen.yaml"
    pipeline = ExperimentPipeline(
        config=CONFIG_PATH,
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
    )
    pipeline.generate()


if __name__ == "__main__":
    # test_new_experiment()
    # test_resume_from_checkpoint()
    test_experiment_generation()
