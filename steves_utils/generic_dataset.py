#! /usr/bin/env python3

from os import replace
import torch
import numpy as np


"""
ds_in is expected to be in the form of
{
    <domain>: {
        <label>: np.ndarray whos shape=(<however many X>, x)
    }
}
"""
def create_dataset(
    ds_in:dict,
    train_val_test_percents:tuple,
    num_examples_per_domain_per_class:int,
    seed:int,
    x_transform_func=lambda x: x,
):
    rng = torch.Generator().manual_seed(seed)
    all_train = []
    all_val   = []
    all_test  = []

    n_train = num_examples_per_domain_per_class*train_val_test_percents[0]
    n_val = num_examples_per_domain_per_class*train_val_test_percents[1]
    n_test = num_examples_per_domain_per_class*train_val_test_percents[2]

    for domain, label_and_x_dict in ds_in.items():
        for label, all_x in label_and_x_dict.items():

            assert(len(all_x) >= num_examples_per_domain_per_class)
            train, val, test = torch.utils.data.random_split(all_x, (n_train, n_val, n_test))

            for e in train: all_train.append( (x_transform_func(e), label, domain) )
            for e in val: all_val.append(     (x_transform_func(e), label, domain) )
            for e in test: all_test.append(   (x_transform_func(e), label, domain) )