#! /usr/bin/env python3
import numpy as np
import pickle
import os
from steves_utils.CORES.utils import (
    node_name_to_id,
    ALL_NODES,
    ALL_DAYS
)

import steves_utils.simple_datasets.episodic_dataloader

from steves_utils.utils_v2 import (norm, get_datasets_base_path)
from steves_utils.simple_datasets.CORES.utils import genericize_stratified_dataset

def get_episodic_dataloaders(
    nodes:list,
    days:list,
    num_examples_per_day_per_node:int,
    seed:int,
    n_shot,
    n_way,
    n_query,
    train_val_test_k_factors,
    normalize_type:str=None,
    pickle_path:str=os.path.join(get_datasets_base_path(), "cores.stratified_ds.2022A.pkl"),
    train_val_test_percents=(0.7,0.15,0.15)
)->tuple:
    with open(pickle_path, "rb") as f:
        stratified_ds_all = pickle.load(f)

    gsd = genericize_stratified_dataset(sds=stratified_ds_all["data"], domains=days, labels=nodes, n_per_u_per_y=num_examples_per_day_per_node)

    if normalize_type != None:
        x_transform_func = lambda x: norm(x, normalize_type)
    else:
        x_transform_func = None

    dataloaders = steves_utils.simple_datasets.episodic_dataloader.get_episodic_dataloaders(
        n_shot=n_shot,
        n_way=n_way,
        n_query=n_query,
        stratified_ds=gsd,
        train_val_test_percents=train_val_test_percents,
        num_examples_per_domain_per_class=num_examples_per_day_per_node,
        seed=seed,
        x_transform_func=x_transform_func,
        train_val_test_k_factors=train_val_test_k_factors,
    )

    return dataloaders


import unittest

def numpy_to_hash(n:np.ndarray):
    return hash(n.data.tobytes())

class Test_Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TRAIN, cls.VAL, cls.TEST = get_episodic_dataloaders(
            nodes=ALL_NODES,
            days=ALL_DAYS,
            num_examples_per_day_per_node=100,
            seed=1337,
            n_shot=1,
            n_way=len(ALL_NODES),
            n_query=1,
            train_val_test_k_factors=(1,1,1)
        )

    def test_all_x_unique(self):
        pass
    
    def test_expected_lens(self):
        pass

    def test_expected_domains(self):
        pass


    def test_expected_labels(self):
        pass


if __name__ == "__main__":
    unittest.main()
