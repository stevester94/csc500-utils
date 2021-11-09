#! /usr/bin/python3

import os

from torch._C import dtype

def get_datasets_base_path():
    return os.environ["DATASETS_ROOT_PATH"]

def get_files_with_suffix_in_dir(path, suffix):
    """Returns full path"""
    (_, _, filenames) = next(os.walk(path))
    return [os.path.join(path,f) for f in filenames if f.endswith(suffix)]

def normalize_val(min, max, val):
    if val < min or val > max:
        raise Exception("Val is out of range")
    return float(val - min) / float(max-min)

def denormalize_val(min, max, val):
    if val > 1.0 or val < 0.0:
        raise Exception("Val is out of range")
    return val * (max-min) + min

def get_past_runs_dir():
    return os.path.join(os.environ["CSC500_ROOT_PATH"], "/mnt/wd500GB/CSC500/csc500-super-repo/csc500-past-runs/")

if __name__ == "__main__":
    import unittest
    import numpy as np
    class test_normalization(unittest.TestCase):
        def test_int_normalization(self):
            MIN = -1000
            MAX = 1000
            COUNT = 1000

            # numbers = np.random.default_rng(1337).choice(list(range(MIN, MAX+1)),COUNT, replace=False)
            numbers = np.arange(MIN, MAX)

            for n in numbers:
                nrm = normalize_val(MIN, MAX, n)
                self.assertTrue(nrm <= 1.0 and nrm >= 0.0)

                dnrm = round(denormalize_val(MIN, MAX, nrm))

                self.assertEqual(n, dnrm)

                print(n, nrm, dnrm)


        def test_float_normalization(self):
            MIN = -1000
            MAX = 1000
            COUNT = 10000

            numbers = np.random.default_rng(1337).uniform(MIN, MAX, COUNT)

            for n in numbers:
                nrm = normalize_val(MIN, MAX, n)
                self.assertTrue(nrm <= 1.0 and nrm >= 0.0)

                dnrm = denormalize_val(MIN, MAX, nrm)

                self.assertAlmostEqual(n, dnrm)

                print(n, nrm, dnrm)
    unittest.main()