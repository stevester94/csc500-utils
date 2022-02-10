#! /usr/bin/env python3

import re
import matplotlib.pyplot as plt
import json
from steves_utils.ptn_train_eval_test_jig import PTN_Train_Eval_Test_Jig
import pandas as pds
import textwrap as twp
import matplotlib.patches as mpatches




def get_jig_diagram(experiment):
    fig, ax = plt.subplots()
    fig.set_size_inches(15,7)

    # Get Loss Curve
    PTN_Train_Eval_Test_Jig.do_diagram(experiment["history"], ax)

    return ax
    

###
# Get Results Table
###
def get_results_table(experiment):
    fig, ax = plt.subplots()
    fig.set_size_inches(15,7)
    ax.set_axis_off() 
    ax.set_title("Results")
    t = ax.table(
        [
            ["Source Val Label Accuracy", "{:.2f}".format(experiment["results"]["source_val_label_accuracy"])],
            ["Source Val Label Loss", "{:.2f}".format(experiment["results"]["source_val_label_loss"])],
            ["Target Val Label Accuracy", "{:.2f}".format(experiment["results"]["target_val_label_accuracy"])],
            ["Target Val Label Loss", "{:.2f}".format(experiment["results"]["target_val_label_loss"])],

            ["Source Test Label Accuracy", "{:.2f}".format(experiment["results"]["source_test_label_accuracy"])],
            ["Source Test Label Loss", "{:.2f}".format(experiment["results"]["source_test_label_loss"])],
            ["Target Test Label Accuracy", "{:.2f}".format(experiment["results"]["target_test_label_accuracy"])],
            ["Target Test Label Loss", "{:.2f}".format(experiment["results"]["target_test_label_loss"])],
            ["Total Epochs Trained", "{:.2f}".format(experiment["results"]["total_epochs_trained"])],
            ["Total Experiment Time Secs", "{:.2f}".format(experiment["results"]["total_experiment_time_secs"])],
        ],
        loc="best",
        cellLoc='left',
        colWidths=[0.3,0.4],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1.5, 2)

    return ax


###
# Get Parameters Table
###
def get_parameters_table(experiment):
    fig, ax = plt.subplots()
    fig.set_size_inches(30,14)
    ax.set_axis_off() 
    ax.set_title("Parameters")

    table_data = [
        ["Experiment Name", experiment["parameters"]["experiment_name"]],
        ["Learning Rate", experiment["parameters"]["lr"]],
        ["Num Epochs", experiment["parameters"]["n_epoch"]],
        ["patience", experiment["parameters"]["patience"]],
        ["seed", experiment["parameters"]["seed"]],
        ["Source Domains", str(experiment["parameters"]["domains_source"])],
        ["Target Domains", str(experiment["parameters"]["domains_target"])],

        ["N per domain per label source", experiment["parameters"]["num_examples_per_domain_per_label_source"]],
        ["N per domain per label target", experiment["parameters"]["num_examples_per_domain_per_label_target"]],

        ["(n_shot, n_way, n_query)", str((experiment["parameters"]["n_shot"], experiment["parameters"]["n_way"],experiment["parameters"]["n_query"]))],
        ["train_k, val_k, test_k", str((experiment["parameters"]["train_k_factor"], experiment["parameters"]["val_k_factor"],experiment["parameters"]["test_k_factor"]))],
        ["Source Labels (n={})".format(len(experiment["parameters"]["labels_source"])),
            experiment["parameters"]["labels_source"]   ],
        ["Target Labels (n={})".format(len(experiment["parameters"]["labels_target"])),
            experiment["parameters"]["labels_target"]   ],

        ["normalize (source,target)", 
            (experiment["parameters"]["normalize_source"], experiment["parameters"]["normalize_target"])],
    ]

    table_data = [(e[0], twp.fill(str(e[1]), 70)) for e in table_data]

    t = ax.table(
        table_data,
        loc="best",
        cellLoc='left',
        colWidths=[0.3,0.45],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1.5, 2)

    # Manually set this rows height (This is the source classes)
    c = t.get_celld()[(11,1)]
    c.set_height( c.get_height() * 3 )
    c.set_fontsize(15)

    # Manually set this rows height (This is the target classes)
    c = t.get_celld()[(12,1)]
    c.set_height( c.get_height() * 3 )
    c.set_fontsize(15)
    
    return ax



    #
    # Build a damn pandas dataframe for the per domain accuracies and plot it
    # 
def get_domain_accuracies(experiment):
    fig, ax = plt.subplots()
    fig.set_size_inches(15,7)
    ax.set_title("Per Domain Validation Accuracy")

    # Convert the dict to a list of tuples
    per_domain_accuracy = experiment["results"]["per_domain_accuracy"]
    per_domain_accuracy = [(domain, v["accuracy"], v["source?"]) for domain,v in per_domain_accuracy.items()]


    df = pds.DataFrame(per_domain_accuracy, columns=["domain", "accuracy", "source?"])
    df.domain = df.domain.astype(int)
    df = df.set_index("domain")
    df = df.sort_values("domain")

    domain_colors = {True: 'r', False: 'b'}
    df['accuracy'].plot(kind='bar', color=[domain_colors[i] for i in df['source?']], ax=ax)

    source_patch = mpatches.Patch(color=domain_colors[True], label='Source Domain')
    target_patch = mpatches.Patch(color=domain_colors[False], label='Target Domain')
    ax.legend(handles=[source_patch, target_patch])
    ax.set_ylim([0.0, 1.0])
    plt.sca(ax)
    plt.xticks(rotation=45, fontsize=13)
    
    return ax
