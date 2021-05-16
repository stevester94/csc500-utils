#! /usr/bin/env python3


import sys
from datetime import datetime

from steves_utils.graphing import plot_confusion_matrix, plot_loss_curve, save_confusion_matrix, save_loss_curve
from steves_utils import utils
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np



ds, cardinality = Simple_ORACLE_Dataset_Factory(
    ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
    runs_to_get=[1],
    distances_to_get=[8],
    serial_numbers_to_get=ALL_SERIAL_NUMBERS[:3]
)

print("Total Examples:", cardinality)
print("That's {}GB of data (at least)".format( cardinality * ORIGINAL_PAPER_SAMPLES_PER_CHUNK * 2 * 8 / 1024 / 1024 / 1024))


for e in ds.batch(1000):
    print(e["serial_number_id"])