#! /usr/bin/env python3
import numpy as np
import pickle
from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    serial_number_to_id
)

from steves_utils.simple_datasets.general_dataset import create_dataset
from steves_utils.utils_v2 import norm

def get_dataset(
    serial_numbers:list,
    distances:list,
    num_examples_per_distance_per_serial:int,
    seed:int,
    pickle_path:str,
    normalize_type:str=None,
    train_val_test_percents=(0.7,0.15,0.15)
):
    d = pickle.load(open(pickle_path, "rb"))

    formed_d = {}

    for distance in distances:
        formed_d[distance] = {}
        for serial in serial_numbers:
            formed_d[distance][serial_number_to_id(serial)] = d[distance][serial][:num_examples_per_distance_per_serial]

    ds = create_dataset(
        ds_in=formed_d,
        train_val_test_percents=train_val_test_percents,
        num_examples_per_domain_per_class=num_examples_per_distance_per_serial,
        seed=seed,
        x_transform_func= lambda x: norm(x, normalize_type) if normalize_type is not None else None
    )

    return ds


if __name__ == "__main__":
    train, val, test = get_dataset(
        serial_numbers=ALL_SERIAL_NUMBERS,
        distances=ALL_DISTANCES_FEET,
        num_examples_per_distance_per_serial=100,
        seed=1337,
        pickle_path="/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/oracle.pkl",
        # normalize_type=None,
        # train_val_test_percents=(0.7,0.15,0.15)
    )

