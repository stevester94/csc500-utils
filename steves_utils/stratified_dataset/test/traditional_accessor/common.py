#! /usr/bin/env python3
import unittest
import numpy as np
from steves_utils.stratified_dataset.traditional_accessor import Traditional_Accessor_Factory

from steves_utils.utils_v2 import to_hash

import torch

class Test_Traditional_Accessor(unittest.TestCase):
    labels = None
    domains = None
    num_examples_per_domain_per_label = None
    pickle_path = None
    seed = 1337
    train_val_test_percents=(0.7,0.15,0.15)
    @classmethod
    def setUpClass(cls):
        taf = Traditional_Accessor_Factory(
            labels=cls.labels,
            domains=cls.domains,
            num_examples_per_domain_per_label=cls.num_examples_per_domain_per_label,
            seed=cls.seed,
            pickle_path=cls.pickle_path,
            train_val_test_percents=cls.train_val_test_percents
        )

        cls.TRAIN, cls.VAL, cls.TEST = (taf.get_train(), taf.get_val(), taf.get_test())


    def test_all_x_unique(self):
        all_h = []

        for ds in (self.TRAIN, self.VAL, self.TEST):
            all_h.extend( [to_hash(x) for (x,y,u) in ds] )

        self.assertEqual(
            len(all_h),
            len(set(all_h))
        )
    
    def test_expected_lens(self):
        n = self.num_examples_per_domain_per_label * len(self.labels) * len(self.domains)

        self.assertAlmostEqual(len(self.TRAIN), n*self.train_val_test_percents[0])
        self.assertAlmostEqual(len(self.VAL), n*self.train_val_test_percents[1])
        self.assertAlmostEqual(len(self.TEST), n*self.train_val_test_percents[2])


    def test_expected_domains(self):
        all_u = []

        for ds in (self.TRAIN, self.VAL, self.TEST):
            all_u.extend( [u for (x,y,u) in ds] )
        
        self.assertEqual(
            set(all_u),
            set(self.domains)
        )


    def test_expected_labels(self):
        all_y = []

        for ds in (self.TRAIN, self.VAL, self.TEST):
            all_y.extend( [y for (x,y,u) in ds] )
        
        self.assertEqual(
            set(all_y),
            set([self.labels.index(y) for y in self.labels])
        )

    def test_change_seed_changes(self):
        taf = Traditional_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            seed=self.seed+1,
            pickle_path=self.pickle_path,
        )

        OLD_TRAIN, OLD_VAL, OLD_TEST = self.TRAIN, self.VAL, self.TEST
        NEW_TRAIN, NEW_VAL, NEW_TEST = (taf.get_train(), taf.get_val(), taf.get_test())

        old_hash = sum([to_hash(x) for x,y,u in OLD_TRAIN + OLD_VAL + OLD_TEST])
        new_hash = sum([to_hash(x) for x,y,u in NEW_TRAIN + NEW_VAL + NEW_TEST])

        self.assertNotEqual(
            old_hash,
            new_hash
        )

    def test_same_seed_no_changes(self):
        taf = Traditional_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            seed=self.seed,
            pickle_path=self.pickle_path,
        )

        OLD_TRAIN, OLD_VAL, OLD_TEST = self.TRAIN, self.VAL, self.TEST
        NEW_TRAIN, NEW_VAL, NEW_TEST = (taf.get_train(), taf.get_val(), taf.get_test())

        old_hash = sum([to_hash(x) for x,y,u in OLD_TRAIN + OLD_VAL + OLD_TEST])
        new_hash = sum([to_hash(x) for x,y,u in NEW_TRAIN + NEW_VAL + NEW_TEST])

        self.assertEqual(
            old_hash,
            new_hash
        )
    
    def test_x_transform(self):
        x_transform = lambda x: x*2

        taf = Traditional_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            seed=self.seed,
            pickle_path=self.pickle_path,
            x_transform_func=x_transform
        )

        OLD = self.TRAIN, self.VAL, self.TEST
        NEW = (taf.get_train(), taf.get_val(), taf.get_test())

        for old_ds, new_ds in zip(OLD, NEW):
            for old_ex, new_ex in zip(old_ds, new_ds):
                self.assertTrue(
                    np.array_equal(old_ex[0]*2, new_ex[0])
                )

    def test_example_transform(self):
        example_transform = lambda ex: (ex[0],ex[1],ex[2]+1337)

        taf = Traditional_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            seed=self.seed,
            pickle_path=self.pickle_path,
            example_transform_func=example_transform
        )

        OLD = self.TRAIN, self.VAL, self.TEST
        NEW = (taf.get_train(), taf.get_val(), taf.get_test())

        def compare_ex(a, b):
            equal = True
            equal = equal and np.array_equal(a[0], b[0])
            equal = equal and a[1] == b[1]
            equal = equal and a[2] == b[2]
            return equal

        for old_ds, new_ds in zip(OLD, NEW):
            for old_ex, new_ex in zip(old_ds, new_ds):
                mod_ex = example_transform(old_ex)
                self.assertTrue(
                    compare_ex(new_ex, mod_ex)
                )
    def test_both_transforms(self):
        example_transform = lambda ex: (ex[0],ex[1],ex[2]+1337)
        x_transform = lambda x: x*2


        taf = Traditional_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            seed=self.seed,
            pickle_path=self.pickle_path,
            example_transform_func=example_transform,
            x_transform_func=x_transform
        )

        OLD = self.TRAIN, self.VAL, self.TEST
        NEW = (taf.get_train(), taf.get_val(), taf.get_test())

        def compare_ex(a, b):
            equal = True
            equal = equal and np.array_equal(a[0], b[0])
            equal = equal and a[1] == b[1]
            equal = equal and a[2] == b[2]
            return equal

        for old_ds, new_ds in zip(OLD, NEW):
            for old_ex, new_ex in zip(old_ds, new_ds):
                new_x = x_transform(old_ex[0])
                mod_ex = example_transform(old_ex)
                mod_ex = (new_x, mod_ex[1], mod_ex[2])

                self.assertTrue(
                    compare_ex(new_ex, mod_ex)
                )


    def test_float32(self):
        old_dtype = torch.get_default_dtype()

        torch.set_default_dtype(torch.float32)
        taf = Traditional_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            seed=self.seed,
            pickle_path=self.pickle_path,
        )
        dtype = next(iter(taf.get_train()))[0].dtype

        self.assertEqual(
            dtype,
            torch.float32
        )

        torch.set_default_dtype(old_dtype)

    def test_float64(self):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        taf = Traditional_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            seed=self.seed,
            pickle_path=self.pickle_path,
        )
        dtype = next(iter(taf.get_train()))[0].dtype

        self.assertEqual(
            dtype,
            torch.float64
        )

        torch.set_default_dtype(old_dtype)

if __name__ == "__main__":
    unittest.main()
