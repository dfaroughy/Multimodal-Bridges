import os
from pipeline.helpers import SimpleLogger as log


log.warnings_off()

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
OUTPUT_PATH = "/home/df630/Multimodal-Bridges/tests/output"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config_continuous.yaml")

def test_databatch():
    # TODO
    pass


if __name__ == "__main__":
    test_databatch()
