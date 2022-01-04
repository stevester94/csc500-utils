#! /usr/bin/env python3


import numpy as np
import pickle
import os
from definitions import *
from utils import get_dataset

class CORES_Sequence:
    def __init__(
        self,
        desired_serial_numbers,
        desired_runs,
        desired_distances,
        window_length,
        window_stride,
        num_examples_per_device,
        seed,
        max_cache_size=1e6, # IDK
        prime_cache=False,
        return_IQ_as_tuple_with_offset=False, # Used for debugging
    ) -> None:
        pass


def load_from_disk(dataset_index):
    root_dir = './'
    if dataset_index == WIFI_PREAM:
        dataset_name = 'grid_2019_12_25.pkl'
    elif dataset_index == WIFI2_PREAM:
        dataset_name = 'grid_2020_02_03.pkl'
    elif dataset_index == WIFI3_PREAM:
        dataset_name = 'grid_2020_02_04.pkl'
    elif dataset_index == WIFI4_PREAM:
        dataset_name = 'grid_2020_02_05.pkl'
    elif dataset_index == WIFI5_PREAM:
        dataset_name = 'grid_2020_02_06.pkl'
        
    else:
        raise ValueError('Wrong dataset index')
    dataset_path = root_dir + dataset_name
    with open(dataset_path,'rb') as f:
        dataset = pickle.load(f)
    return dataset


if __name__ == "__main__":
    pass