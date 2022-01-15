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
    iterator_seed:int,
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
        iterator_seed=iterator_seed,
        x_transform_func=x_transform_func,
        train_val_test_k_factors=train_val_test_k_factors,
    )

    return dataloaders


import unittest

def numpy_to_hash(n:np.ndarray):
    return hash(n.data.tobytes())

from steves_utils.simple_datasets.episodic_test_cases import(
    test_correct_domains,
    test_correct_labels,
    test_correct_example_count_per_domain_per_label,
    test_dls_disjoint,
    test_dls_equal,
    test_dls_notEqual,
    test_len,
    test_splits,
    test_episodes_have_no_repeats,
    test_normalization,
    test_shape,
    test_approximate_number_episodes
)

class Test_Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.desired_domains = ALL_DAYS
        cls.desired_labels  = ALL_NODES
        cls.num_examples_per_domain_per_label=100
        cls.desired_seed=1337
        cls.desired_n_shot=2
        cls.desired_n_way=len(ALL_NODES)
        cls.desired_n_query=2
        cls.desired_train_val_test_k_factors=(1,1,1)
        cls.train_val_test_percents=(0.7,0.15,0.15)

        cls.TRAIN, cls.VAL, cls.TEST = get_episodic_dataloaders(
            days=cls.desired_domains,
            nodes=cls.desired_labels,
            num_examples_per_day_per_node=cls.num_examples_per_domain_per_label,
            iterator_seed=cls.desired_seed,
            n_shot=cls.desired_n_shot,
            n_way=cls.desired_n_way,
            n_query=cls.desired_n_query,
            train_val_test_k_factors=cls.desired_train_val_test_k_factors,
            train_val_test_percents=cls.train_val_test_percents
        )

        cls.ALL_DL = (cls.TRAIN, cls.VAL, cls.TEST)

        cls.generic_labels = [node_name_to_id(y) for y in cls.desired_labels]

    def test_correct_domains(self):
        for dl in self.ALL_DL:
            test_correct_domains(self, dl, self.desired_domains)

    def test_correct_labels(self):
        for dl in self.ALL_DL:
            test_correct_labels(self, dl, self.generic_labels)


    def test_correct_example_count_per_domain_per_label(self):
        print("domains", len(self.desired_domains))
        print("labels", len(self.desired_labels))
        for dl,ratio in zip(self.ALL_DL[:1], self.train_val_test_percents[:1]):
            test_correct_example_count_per_domain_per_label(self, dl, int(self.num_examples_per_domain_per_label*ratio))


    def test_dls_disjoint(self):
        test_dls_disjoint(self, self.ALL_DL)

    
    def test_shape(self):
        for dl in self.ALL_DL:
            test_shape(self, dl, self.desired_n_way, self.desired_n_shot, self.desired_n_query)

    def test_repeatability(self):
        TRAIN, VAL, TEST = get_episodic_dataloaders(
                    days=self.desired_domains,
                    nodes=self.desired_labels,
                    num_examples_per_day_per_node=self.num_examples_per_domain_per_label,
                    iterator_seed=self.desired_seed,
                    n_shot=self.desired_n_shot,
                    n_way=self.desired_n_way,
                    n_query=self.desired_n_query,
                    train_val_test_k_factors=self.desired_train_val_test_k_factors,
                )
        
        for a,b in zip(self.ALL_DL, (TRAIN, VAL, TEST)):
            test_dls_equal(self, a,b)

    def test_approximate_number_episodes(self):
        print("ALL_DAYS", len(ALL_DAYS))
        print("ALL_NODES", len(ALL_NODES))
        for i,dl in enumerate(self.ALL_DL):
            if i == 0: continue
            test_approximate_number_episodes(
                self,
                dl,
                self.desired_train_val_test_k_factors[i],
                int(self.num_examples_per_domain_per_label*self.train_val_test_percents[i]),
                len(self.desired_labels),
                len(self.desired_domains),
                self.desired_n_way,
                self.desired_n_shot,
                self.desired_n_query,
            )
    
    def test_len(self):
        for dl in self.ALL_DL:
            test_len(self, dl)

    def test_episodes_have_no_repeats(self):
        for i,dl in enumerate(self.ALL_DL):
            test_episodes_have_no_repeats(self, dl)
    
    def test_splits(self):
        test_splits(self, self.ALL_DL, self.train_val_test_percents)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "limited":
        suite = unittest.TestSuite()

        # suite.addTest(Test_Dataset("test_approximate_number_episodes"))
        suite.addTest(Test_Dataset("test_splits"))

        runner = unittest.TextTestRunner()
        runner.run(suite)
    elif len(sys.argv) > 1:
        Test_Dataset().test_reproducability()
    else:
        unittest.main()