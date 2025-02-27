import json
import os
import warnings


def get_from_json(key, path, name="metadata.json"):
    path = os.path.join(path, name)
    with open(path, "r") as f:
        file = json.load(f)
    return file[key]


class SimpleLogger:
    @staticmethod
    def info(message, condition=True):
        if condition:
            print("\033[94m\033[1mINFO: \033[0m\033[00m", message)
        return

    @staticmethod
    def warn(message, condition=True):
        if condition:
            print("\033[31m\033[1mWARNING: \033[0m\033[00m", message)
        return

    @staticmethod
    def warnings_off():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)