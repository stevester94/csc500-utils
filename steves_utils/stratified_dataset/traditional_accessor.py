#! /usr/bin/env python3
from typing import Tuple
import numpy as np
import math
import torch

from steves_utils.stratified_dataset.stratified_dataset import Stratified_Dataset
from steves_utils.stratified_dataset.utils import filter_sds_in_place



def condense_and_split_sds(
    sds:Stratified_Dataset,
    train_val_test_percents:tuple,
    num_examples_per_domain_per_class:int,
    seed:int,
    x_transform_func=None,
    example_transform_func=None
)->Tuple[list,list,list]:
    """Squash an SDS and split into train, val, test after applying lambda funcs

    for each domain,
        for each label
            X is shuffled, split into the appropriate percentages, x_transformed, example_transformed
    train,val,test are then shuffled
    """

    all_train = []
    all_val   = []
    all_test  = []

    n_train = math.floor(num_examples_per_domain_per_class*train_val_test_percents[0])
    n_val = math.floor(num_examples_per_domain_per_class*train_val_test_percents[1])
    n_test = (num_examples_per_domain_per_class - n_train - n_val)

    rng = np.random.default_rng(seed)

    if x_transform_func == None:
        x_transform_func = lambda x: x

    if example_transform_func == None:
        example_transform_func = lambda ex: ex

    for u, y_X_dict in sds.get_data().items():
        for y, X in y_X_dict.items():

            rng.shuffle(X)

            if len(X) < num_examples_per_domain_per_class:
                raise RuntimeError("Number of examples requested for (u={}, y={}) is too high, have only {} but wanted {}".format(
                    u, y,
                    len(X), num_examples_per_domain_per_class
                ))

            train = X[:n_train]
            val   = X[n_train:n_train+n_val]
            test  = X[n_train+n_val:n_train+n_val+n_test]

            for f in train:
                e = torch.from_numpy(f)
                e = e.to(torch.get_default_dtype())
                all_train.append( example_transform_func( (x_transform_func(e), y, u) ))

            for f in val:
                e = torch.from_numpy(f)
                e = e.to(torch.get_default_dtype())
                all_val.append(     example_transform_func( (x_transform_func(e), y, u) ))

            for f in test:
                e = torch.from_numpy(f)
                e = e.to(torch.get_default_dtype())
                all_test.append(   example_transform_func( (x_transform_func(e), y, u) ))
    
    # Done in place
    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    return all_train, all_val, all_test

class Traditional_Accessor_Factory:
    """Returns train, val, test as lists in form of (x,y,u)

    train, val, test are all thoroughly shuffled based on <seed>
    x_transform_func is first applied, then example_transform_func
    """
    def __init__(
        self,
        labels:list,
        domains:list,
        num_examples_per_domain_per_label:int,
        pickle_path:str,
        seed:int,
        x_transform_func=None,
        example_transform_func=None,
        train_val_test_percents=(0.7,0.15,0.15)
    ) -> None:
        self.sds = Stratified_Dataset(pickle_path)

        filter_sds_in_place(sds=self.sds,
            domains=domains,
            labels=labels,
            num_examples_per_domain_per_label=num_examples_per_domain_per_label,
            seed=seed
        )

        self.train, self.val, self.test = condense_and_split_sds(
            sds=self.sds,
            train_val_test_percents=train_val_test_percents,
            num_examples_per_domain_per_class=num_examples_per_domain_per_label,
            x_transform_func=x_transform_func,
            example_transform_func=example_transform_func,
            seed=seed
        )

        del self.sds

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

    taf = Traditional_Accessor_Factory(
        labels=ALL_NODES,
        domains=ALL_DAYS,
        num_examples_per_domain_per_label=100,
        pickle_path=os.path.join(get_datasets_base_path(), "cores.stratified_ds.2022A.pkl"),
        seed=1337
    )

    train = taf.get_train()
    val = taf.get_val()
    test = taf.get_test()

    for x,y,u in train:
        print(y,u)