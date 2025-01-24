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

def test_databatch():
    # TODO
    pass


if __name__ == "__main__":
    test_databatch()
