#! /usr/bin/env python3
import unittest
import os

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
)

from steves_utils.utils_v2 import get_datasets_base_path

from steves_utils.stratified_dataset.test.traditional_accessor.common import Test_Traditional_Accessor


Test_Traditional_Accessor.labels = ALL_SERIAL_NUMBERS
Test_Traditional_Accessor.domains = ALL_DISTANCES_FEET
Test_Traditional_Accessor.num_examples_per_domain_per_label = 100
Test_Traditional_Accessor.pickle_path = os.path.join(get_datasets_base_path(), "oracle.stratified_ds.2022A.pkl")


unittest.main()