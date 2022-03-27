import subprocess
import json
import os
from steves_utils.utils_v2 import get_experiments_base_path
import sys




def parse_results_from_ipynb_dict(d:dict):
    for cell in d["cells"]:
        if "tags" in cell["metadata"] and cell["metadata"]["tags"] == ["experiment_json"]:
            if len(cell["outputs"]) != 1:
                print(type(cell["outputs"]))
                print(cell["outputs"])
                l  = len(cell["outputs"])
                raise Exception(f"len of experiment_json unexpected length (got {l})")
            if len(cell["outputs"][0]["data"]["text/plain"]) != 1:
                raise Exception("len of cell data unexpected length")

            break

    # The string itself is surrounded by single quotes. I'm being lazy here and just eval'ing it
    experiment_json = eval(cell["outputs"][0]["data"]["text/plain"][0])
    experiment = json.loads(experiment_json)

    return experiment

def get_experiments_from_path(start_path):
    experiment_dot_json_paths = subprocess.getoutput('find {} | grep trial.ipynb'.format(start_path))

    experiment_dot_json_paths = experiment_dot_json_paths.split('\n')
    experiments = []

    for p in experiment_dot_json_paths:
        with open(p) as f:
            try:
                experiments.append(
                    parse_results_from_ipynb_dict(json.load(f))
                )
            except:
                print("Error parsing",p, file=sys.stderr)
                raise

    
    return experiments

