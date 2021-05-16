#! /usr/bin/env python3

from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_DISTANCES_FEET, ALL_RUNS, ALL_SERIAL_NUMBERS
from steves_utils.ORACLE.dataset_shuffler import Dataset_Shuffler
import tensorflow as tf
from typing import List
from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS
from steves_utils.utils import get_all_in_dir
import unittest
from shutil import rmtree
import os

SCRATCH_DIR = "/mnt/wd500GB/derp/"



def clear_scrath_dir():
    for thing in get_all_in_dir(SCRATCH_DIR):
        rmtree(thing)

class Test_oracle_dataset_shuffler_safety_features(unittest.TestCase):
    @unittest.expectedFailure
    def test_too_few_piles(self):
        shuffler = Dataset_Shuffler(
            num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
            output_batch_size=1000,
            num_piles=1,
            output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
            output_max_file_size_MB=200,
            # output_max_file_size_MB=1,
            pile_dir=os.path.join(SCRATCH_DIR, "piles"),
            output_dir=os.path.join(SCRATCH_DIR, "output"),
            seed=1337,
        )

    @unittest.expectedFailure
    def test_non_empty_dirs(self):
        clear_scrath_dir()
        os.mkdir(os.path.join(SCRATCH_DIR, "piles"))
        with open(os.path.join(SCRATCH_DIR, "piles", "out"), "w") as f:
            f.write("lol")
        shuffler = Dataset_Shuffler(
            num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
            output_batch_size=1000,
            num_piles=1,
            output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
            output_max_file_size_MB=1,
            # output_max_file_size_MB=1,
            pile_dir=os.path.join(SCRATCH_DIR, "piles"),
            output_dir=os.path.join(SCRATCH_DIR, "output"),
            seed=1337,
            runs_to_get=[1],
            distances_to_get=[8],
            serial_numbers_to_get=[ALL_SERIAL_NUMBERS[0]]
        )

        shuffler.create_and_check_dirs()


    # def test_too_few_piles(self):
        # clear_scrath_dir()



if __name__ == "__main__":
    unittest.main()

        # shuffler = Dataset_Shuffler(
        #     num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
        #     output_batch_size=1000,
        #     num_piles=1,
        #     output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
        #     output_max_file_size_MB=200,
        #     # output_max_file_size_MB=1,
        #     pile_dir=os.path.join(SCRATCH_DIR, "piles"),
        #     output_dir=os.path.join(SCRATCH_DIR, "output"),
        #     seed=1337,
        #     runs_to_get=[1],
        #     distances_to_get=[8],
        #     serial_numbers_to_get=[ALL_SERIAL_NUMBERS[0]]
        # )

    # shuffler = Dataset_Shuffler(
    #     num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
    #     output_batch_size=1000,
    #     num_piles=5,
    #     output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
    #     output_max_file_size_MB=200,
    #     # output_max_file_size_MB=1,
    #     pile_dir="/mnt/wd500GB/derp/pile",
    #     output_dir="/mnt/wd500GB/derp/output",
    #     seed=1337,
    #     runs_to_get=[1],
    #     distances_to_get=[8],
    #     serial_numbers_to_get=[ALL_SERIAL_NUMBERS[0]]
    # )


    # shuffler.create_and_check_dirs()
    # print("Write piles")
    # shuffler.write_piles()
    # print("shuffle")
    # shuffler.shuffle_piles()