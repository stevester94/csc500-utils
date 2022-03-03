import subprocess
import pandas as pd
import json


def parse_results_from_ipynb_dict(d:dict):
    for cell in d["cells"]:
        if "tags" in cell["metadata"] and cell["metadata"]["tags"] == ["experiment_json"]:
            assert len(cell["outputs"]) == 1
            assert len(cell["outputs"][0]["data"]["text/plain"]) == 1

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
            experiments.append(
                parse_results_from_ipynb_dict(json.load(f))
            )
    
    return experiments
