#! /usr/bin/env python3
from abc import ABC, abstractmethod
import unittest

from steves_utils.utils_v2 import to_hash
import pickle
import os

class Stratified_Dataset_Builder(ABC):
    """Abstract base class for classes building stratified datasets

    By making this an ABC to all stratified datasets, we make testing
    easier (we can reuse test code).

    See stratified_dataset/README for more info

    See csc500-dataset-preprocessor/CORES for examples
    """
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def build_dataset(
        seed:int,
        domains:list,
        labels:list,
        out_path:str
    ):
        pass

class Stratified_Dataset_Builder_Basic_Test(unittest.TestCase):
    """Universal test code for basic testing of Stratified Dataset Builders

    Override the below vars and then execute unittest.main()

    See csc500-dataset-preprocessor/CORES for examples
    """
    SDB = "override this with a stratified_dataset_builder object to be tested"
    TEST_DOMAINS = ["some domains"]
    TEST_LABELS  = ["some labels"]

    # Override if the X shape is known and consistent and should be tested
    # Leave None if you don't want to test
    EXPECTED_X_SHAPE = None

    OUT_PATH = "/tmp/sdb.pkl"

    @classmethod
    def setUpClass(cls):
        cls.seed = 1337
        cls.SDB.build_dataset(
            seed=cls.seed,
            domains=cls.TEST_DOMAINS,
            labels=cls.TEST_LABELS,
            out_path=cls.OUT_PATH
        )

        with open(cls.OUT_PATH, "rb") as f:
            cls.d = pickle.load(f)

    # @unittest.SkipTest
    def test_unique_hashes(self):
        data = self.d["data"]
        metadata = self.d["metadata"]
        h = []

        for u, y_X_dict in data.items():
            for y,X in y_X_dict.items():
                for x in X:
                    h.append(
                        to_hash(x)
                    )
        
        self.assertEqual(
            len(h),
            len(set(h))
        )

    # @unittest.SkipTest
    def test_desired_domains(self):
        data = self.d["data"]
        metadata = self.d["metadata"]
        h = []

        self.assertEqual(
            set(self.TEST_DOMAINS),
            set(data.keys())
        )

    # @unittest.SkipTest
    def test_desired_labels(self):
        data = self.d["data"]
        metadata = self.d["metadata"]
        labels = set()

        for u, y_X_dict in data.items():
            for y,X in y_X_dict.items():
                for x in X:
                    labels.add(y)
        
        self.assertEqual(
            set(self.TEST_LABELS),
            labels
        )

    # @unittest.SkipTest
    def test_num_examples_and_shape(self):
        if self.EXPECTED_X_SHAPE is None:
            print("Skipping test_num_examples_and_shape")
            self.skipTest()

        data = self.d["data"]
        metadata = self.d["metadata"]
        h = []

        for u, y_X_dict in data.items():
            for y,X in y_X_dict.items():
                for x in X:
                    self.assertEqual(
                        x.shape,
                        self.EXPECTED_X_SHAPE
                    )

    # @unittest.SkipTest
    def test_seed_randomizes(self):
        self.SDB.build_dataset(
            seed=self.seed+1,
            domains=self.TEST_DOMAINS,
            labels=self.TEST_LABELS,
            out_path=self.OUT_PATH+"2"
        )

        with open(self.OUT_PATH+"2", "rb") as f:
            new_d = pickle.load(f)
        os.unlink(self.OUT_PATH+"2")

        new_d = new_d["data"]
        old_d = self.d["data"]

        for u in new_d.keys():
            for y in new_d[u].keys():
                
                old_domain_label_hashes = []
                new_domain_label_hashes = []

                for old_x, new_x in zip(old_d[u][y], new_d[u][y]):
                    old_domain_label_hashes.append(to_hash(old_x))
                    new_domain_label_hashes.append(to_hash(new_x))
                self.assertNotEqual(
                    tuple(old_domain_label_hashes),
                    tuple(new_domain_label_hashes),
                )