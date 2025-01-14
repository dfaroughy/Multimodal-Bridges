import json
import os


def get_from_json(key, path, name="metadata.json"):
    path = os.path.join(path, name)
    with open(path, "r") as f:
        file = json.load(f)
    return file[key]


class SimpleLogger:
    @staticmethod
    def info(message):
        print("\033[94m\033[1mINFO: \033[0m\033[00m", message)
        return

    @staticmethod
    def warn(message):
        print("\033[31m\033[1mWARNING: \033[0m\033[00m", message)
        return
