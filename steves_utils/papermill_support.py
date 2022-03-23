import os
import papermill


"""
Run a list of experiments using the papermill library to inject the experiment parameters.

Slightly confusing. We rely on relative paths, relative to where the experiment is being run,
for some experiment parameters such as where to write the best model.
This is an issue, because papermill does not change the current working directory.

We change directory to each experiment dir when 
running since papermill uses the cwd, not where the notebook being executed lives.
We save the original cwd of the script so we can cd back up before cd'ing into the next
experiment.

It's basically pushd and popd
"""

def run_trials_with_papermill(
    trials:list,
    notebook_template_path:str,
    notebook_out_name:str,
    trials_dir_path:str,
    best_model_path:str,
    save_best_model:bool,
    )->None:
    script_original_cwd = os.getcwd()

    if not os.path.isdir(trials_dir_path):
        os.mkdir(trials_dir_path)
    else:
        print("Trials dir exists, continuing")

    for i, e in enumerate(trials):
        print(f"Running trial {i}")

        os.chdir(script_original_cwd)

        trial_path = os.path.join(
            trials_dir_path,
            f"{i}"
        )
        if not os.path.isdir(trial_path):
            os.mkdir(trial_path)
        else:
            print(f"Trial {trial_path} exists, skipping")
            continue

        os.chdir(trial_path)

        papermill.execute_notebook(
            notebook_template_path,
            notebook_out_name,
            parameters = {"parameters": e}
        )

        if not save_best_model:
            os.remove(best_model_path)