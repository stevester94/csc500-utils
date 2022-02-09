#! /usr/bin/env python3
import random
import torch
import numpy as np
import pickle
import os
from steves_utils.CORES.utils import (
    node_name_to_id,
    ALL_NODES,
    ALL_DAYS
)

from steves_utils.stratified_dataset.stratified_dataset import Stratified_Dataset
from steves_utils.stratified_dataset.utils import filter_sds_in_place
from steves_utils.stratified_dataset.episodic_dataloader import get_episodic_dataloaders

from steves_utils.utils_v2 import get_datasets_base_path

class Episodic_Accessor_Factory:
    def __init__(
        self,
        labels:list,
        domains:list,
        num_examples_per_domain_per_label:int,
        pickle_path:str,
        dataset_seed:int,
        n_shot:int,
        n_way:int,
        n_query:int,
        iterator_seed:int,
        train_val_test_k_factors,
        x_transform_func=None,
        example_transform_func=None,
        train_val_test_percents=(0.7,0.15,0.15)
    ) -> None:

        if example_transform_func is not None:
            raise Exception("example_transform_func is not implemented")

        sds = Stratified_Dataset(pickle_path)

        
        filter_sds_in_place(sds=sds,
            domains=domains,
            labels=labels,
            num_examples_per_domain_per_label=num_examples_per_domain_per_label,
            seed=dataset_seed
        )

        self.train, self.val, self.test = get_episodic_dataloaders(
            n_shot=n_shot,
            n_way=n_way,
            n_query=n_query,
            sds=sds,
            train_val_test_percents=train_val_test_percents,
            num_examples_per_domain_per_label=num_examples_per_domain_per_label,
            iterator_seed=iterator_seed,
            x_transform_func=x_transform_func,
            train_val_test_k_factors=train_val_test_k_factors,
        )

        del sds

    def get_train(self):
        return self.train
    def get_val(self):
        return self.val
    def get_test(self):
        return self.test

if __name__ == "__main__":
    import os

    from steves_utils.CORES.utils import (
        ALL_DAYS,
        ALL_NODES ,
    )
    from steves_utils.utils_v2 import get_datasets_base_path

    eaf = Episodic_Accessor_Factory(
        labels=ALL_NODES,
        domains=ALL_DAYS,
        num_examples_per_domain_per_label=100,
        pickle_path=os.path.join(get_datasets_base_path(), "cores.stratified_ds.2022A.pkl"),
        dataset_seed=1337,
        n_shot=2,
        n_way=len(ALL_NODES),
        n_query=2,
        iterator_seed=420,
        train_val_test_k_factors=(1,1,1),
        x_transform_func=None,
        example_transform_func=None,
        train_val_test_percents=(0.7,0.15,0.15)
    )

    train = eaf.get_train()
    val = eaf.get_val()
    test = eaf.get_test()

    for u, (support_x, support_y, query_x, query_y, true_y) in train:
        print(u)