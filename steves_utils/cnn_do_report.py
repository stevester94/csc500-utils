#! /usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pds
import textwrap as twp
import matplotlib.patches as mpatches
from pexpect import ExceptionPexpect
from steves_utils.utils_v2 import do_graph



def get_loss_curve(experiment):
    fig, ax = plt.subplots()
    fig.set_size_inches(15,7)

    history = experiment["history"]

    # Bottom Right: src train label vs  src val label
    graphs = [
        {
            "x": history["epoch_indices"],
            "y": history["train_label_loss"],
            "x_label": None,
            "y_label": "Train Label Loss",
            "x_units": "Epoch",
            "y_units": None,
        }, 
        {
            "x": history["epoch_indices"],
            "y": history["source_val_label_loss"],
            "x_label": None,
            "y_label": "Source Val Label Loss",
            "x_units": "Epoch",
            "y_units": None,
        },
        {
            "x": history["epoch_indices"],
            "y": history["target_val_label_loss"],
            "x_label": None,
            "y_label": "Target Val Label Loss",
            "x_units": "Epoch",
            "y_units": None,
        }, 
        {
            "x": history["epoch_indices"],
            "y": history["target_val_acc_label"],
            "x_label": None,
            "y_label": "Target Val Label Accuracy",
            "x_units": "Epoch",
            "y_units": None,
        }, 
        {
            "x": history["epoch_indices"],
            "y": history["source_val_acc_label"],
            "x_label": None,
            "y_label": "Source Val Label Accuracy",
            "x_units": "Epoch",
            "y_units": None,
        }, 
    ]
    do_graph(ax, "Source Train Label Loss vs Source Val Label Loss", graphs)

    return ax

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

def get_parameters_table(experiment):
    fig, ax = plt.subplots()
    fig.set_size_inches(15,7)
    
    ax.set_axis_off() 
    ax.set_title("Parameters")

    table_data = [
            ["Experiment Name", experiment["parameters"]["experiment_name"]],
            ["Learning Rate", experiment["parameters"]["lr"]],
            ["Num Epochs", experiment["parameters"]["n_epoch"]],
            ["Batch Size", experiment["parameters"]["batch_size"]],
            ["patience", experiment["parameters"]["patience"]],
            ["seed", experiment["parameters"]["seed"]],
            ["device", experiment["parameters"]["device"]],
            ["Source Domains", str(experiment["parameters"]["source_domains"])],
            ["Target Domains", str(experiment["parameters"]["target_domains"])],
            ["N per class per domain", str(experiment["parameters"]["num_examples_per_class_per_domain"])],
            ["Classes (n={})".format(len(experiment["parameters"]["desired_classes"])),
                experiment["parameters"]["desired_classes"]   ],
            ["normalize source", experiment["parameters"]["normalize_source"]],
            ["normalize target", experiment["parameters"]["normalize_target"]],
        ]

    table_data = [(e[0], twp.fill(str(e[1]), 70)) for e in table_data]

    t = ax.table(
        table_data,
        loc="best",
        cellLoc='left',
        colWidths=[0.2,0.55],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1.5, 2)

    c = t.get_celld()[(10,1)]
    c.set_height( c.get_height() *5 )
    c.set_fontsize(15)

    return ax

def get_domain_accuracies(experiment):
    fig, ax = plt.subplots()
    fig.set_size_inches(15,7)

    ax.set_title("Per Domain Accuracy")

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