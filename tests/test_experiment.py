import os
import pytest
from pathlib import Path
from utils.experiment_pipeline import ExperimentPipeline

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config.yaml")

def test_experiment_train():
    pipeline = ExperimentPipeline(
        config=CONFIG_PATH,
        accelerator="gpu",
        strategy="auto",
        devices=[0],
    )
    pipeline.train()

def test_experiment_train_from_checkpoint():
    EXP_PATH = f"{RESOURCE_PATH}/output/multimodal-jets/e4941ebe0024468d988baf191943544e"
    pipeline = ExperimentPipeline(
        experiment_path=EXP_PATH,
        accelerator="gpu",
        strategy="auto",
        devices=[0],
        config={"train": 
                       {"max_epochs": 300}
                },
    )
    pipeline.train()

def test_experiment_generation():
    EXP_PATH = f"{RESOURCE_PATH}/output/multimodal-jets/87218a2fba09445e9fe0c61e7386f639"
    pipeline = ExperimentPipeline(
        config=f"{RESOURCE_PATH}/test_config_gen.yaml",
        experiment_path=EXP_PATH,
        accelerator="gpu",
        strategy="auto",
        devices=[0],
    )
    pipeline.generate()


if __name__ == "__main__":
    test_experiment_train()
    # test_experiment_train_from_checkpoint()
    # test_experiment_generation()