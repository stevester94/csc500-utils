#! /usr/bin/env python3
import unittest
import os

from steves_utils.CORES.utils import (
    ALL_DAYS,
    ALL_NODES ,
)

from steves_utils.utils_v2 import get_datasets_base_path

from steves_utils.stratified_dataset.test.traditional_accessor.common import Test_Traditional_Accessor


Test_Traditional_Accessor.labels = ALL_NODES
Test_Traditional_Accessor.domains = ALL_DAYS
Test_Traditional_Accessor.num_examples_per_domain_per_label = 100
Test_Traditional_Accessor.pickle_path = os.path.join(get_datasets_base_path(), "cores.stratified_ds.2022A.pkl")


unittest.main()