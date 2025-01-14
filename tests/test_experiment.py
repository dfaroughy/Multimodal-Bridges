import os
import pytest
from utils.experiment_pipeline import ExperimentPipeline
import shutil

RES_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
OUTPUT_PATH = "/home/df630/Multimodal-Bridges/tests/output"


def test_new_experiment():
    CONFIG_PATH = os.path.join(RES_PATH, "config.yaml")
    new_exp = ExperimentPipeline(config=CONFIG_PATH)
    new_exp.train()
    # shutil.rmtree(new_exp.trainer.config.path)


def test_resume_from_checkpoint():
    EXP_PATH = f"{OUTPUT_PATH}/multimodal-jets/c703493ef98441a690a4591748957abb"
    resume_exp = ExperimentPipeline(
        config={"trainer": {"max_epochs": 45}},
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt",
    )
    resume_exp.train()


def test_experiment_generation():
    EXP_PATH = f"{OUTPUT_PATH}/multimodal-jets/c703493ef98441a690a4591748957abb"
    CONFIG_PATH = f"{RES_PATH}/config_gen.yaml"
    pipeline = ExperimentPipeline(
        config=CONFIG_PATH,
        experiment_path=EXP_PATH,
        load_ckpt="last.ckpt"
    )
    pipeline.generate()


if __name__ == "__main__":
    test_new_experiment()
    # test_resume_from_checkpoint()
    # test_experiment_generation()
