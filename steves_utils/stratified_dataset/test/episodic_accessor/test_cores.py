#! /usr/bin/env python3
import unittest
import os

from steves_utils.CORES.utils import (
    ALL_DAYS,
    ALL_NODES ,
)

from steves_utils.utils_v2 import get_datasets_base_path

from steves_utils.stratified_dataset.test.episodic_accessor.common import Test_Episodic_Accessor


Test_Episodic_Accessor.domains = ALL_DAYS
Test_Episodic_Accessor.labels  = ALL_NODES
Test_Episodic_Accessor.num_examples_per_domain_per_label=100
Test_Episodic_Accessor.dataset_seed=1337
Test_Episodic_Accessor.iterator_seed=420
Test_Episodic_Accessor.n_shot=2
Test_Episodic_Accessor.n_way=len(ALL_NODES)
Test_Episodic_Accessor.n_query=2
Test_Episodic_Accessor.train_val_test_k_factors=(1,2,2)
Test_Episodic_Accessor.train_val_test_percents=(0.7,0.15,0.15)
Test_Episodic_Accessor.pickle_path = os.path.join(get_datasets_base_path(), "cores.stratified_ds.2022A.pkl")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "limited":
        suite = unittest.TestSuite()

        # suite.addTest(Test_Dataset("test_approximate_number_episodes"))
        # suite.addTest(Test_Dataset("test_correct_example_count_per_domain_per_label"))
        suite.addTest(Test_Episodic_Accessor("test_val_and_test_dont_change_between_iterations"))
        
        # suite.addTest(Test_Dataset("test_no_duplicates_in_dl"))

        runner = unittest.TextTestRunner()
        runner.run(suite)
    elif len(sys.argv) > 1:
        Test_Episodic_Accessor().test_reproducability()
    else:
        unittest.main()