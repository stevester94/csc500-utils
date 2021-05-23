#! /usr/bin/env python3

from numpy.core.fromnumeric import sort
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_DISTANCES_FEET, ALL_RUNS, ALL_SERIAL_NUMBERS, get_chunk_of_IQ_based_on_metadata_and_index
from steves_utils.ORACLE.dataset_shuffler import Dataset_Shuffler
import tensorflow as tf
from typing import List
from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS
from steves_utils.utils import get_all_in_dir
import unittest
from shutil import rmtree
import os
from steves_utils.ORACLE.windowed_dataset_shuffler import Windowed_Dataset_Shuffler
from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory
import numpy as np
import copy
import math

SCRATCH_DIR = "/mnt/wd500GB/derp/"



def clear_scratch_dir():
    for thing in get_all_in_dir(SCRATCH_DIR):
        rmtree(thing)


class Test_Shuffler_Datasets_Only(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.num_windowed_examples_per_device = int(3e3)
        self.num_val_examples_per_device = int(1e3)
        self.num_test_examples_per_device = int(2e3)

        self.expected_train_count = self.num_windowed_examples_per_device * len(ALL_SERIAL_NUMBERS)
        self.expected_val_count = self.num_val_examples_per_device * len(ALL_SERIAL_NUMBERS)
        self.expected_test_count = self.num_test_examples_per_device * len(ALL_SERIAL_NUMBERS)

        input_shuffled_ds_num_samples_per_chunk=4*ORIGINAL_PAPER_SAMPLES_PER_CHUNK
        output_window_size=ORIGINAL_PAPER_SAMPLES_PER_CHUNK
        stride_length=1

        self.expected_train_replication_factor = math.ceil((input_shuffled_ds_num_samples_per_chunk - output_window_size)/stride_length + 1)
        self.expected_val_replication_factor = math.ceil(input_shuffled_ds_num_samples_per_chunk/output_window_size)
        self.expected_test_replication_factor = math.ceil(input_shuffled_ds_num_samples_per_chunk/output_window_size)

        self.shuffler = Windowed_Dataset_Shuffler(
            input_shuffled_ds_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/all_shuffled_chunk-512/output",
            input_shuffled_ds_num_samples_per_chunk=input_shuffled_ds_num_samples_per_chunk,
            output_batch_size=100,
            seed=1337,
            num_windowed_examples_per_device=int(200e3),
            num_val_examples_per_device=int(10e3),
            num_test_examples_per_device=int(50e3),
            output_max_file_size_MB=100,
            output_window_size=output_window_size, 
            serials_to_filter_on=ALL_SERIAL_NUMBERS,
            working_dir=SCRATCH_DIR,
            output_format_str="batch-{batch_size}_part-{part}.tfrecord_ds",
            stride_length=stride_length
        )

    @unittest.skip("Skip cardinality to save time")
    def test_dataset_cardinality(self):
        acceptable_cardinality_delta_percent = 0.05
        
        datasets = self.shuffler.get_datasets()

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"]   

        train_count = 0
        for e in train_ds:
            train_count += e["index_in_file"].shape[0]

        val_count = 0
        for e in val_ds:
            val_count += e["index_in_file"].shape[0]

        test_count = 0
        for e in test_ds:
            test_count += e["index_in_file"].shape[0]


        self.assertAlmostEqual(
            train_count, 
            self.expected_train_count, 
            delta=self.expected_train_count*acceptable_cardinality_delta_percent
        )
        self.assertAlmostEqual(
            val_count,
            self.expected_val_count,
            delta=self.expected_val_count*acceptable_cardinality_delta_percent
        )
        self.assertAlmostEqual(
            test_count,
            self.expected_test_count,
            delta=self.expected_test_count*acceptable_cardinality_delta_percent
        )
        
        self.assertGreaterEqual(
            train_count, 
            self.expected_train_count, 
        )
        self.assertGreaterEqual(
            val_count,
            self.expected_val_count,
        )
        self.assertGreaterEqual(
            test_count,
            self.expected_test_count,
        )

    def test_dataset_shape(self):
        datasets = self.shuffler.get_datasets()

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"]  

        for e in train_ds.take(1):
            print(e)
# class Test_shuffler_end_to_end(unittest.TestCase):
#     @classmethod
#     def setUpClass(self):
#         self.num_windowed_examples_per_device = int(3e3)
#         self.num_val_examples_per_device = int(1e3)
#         self.num_test_examples_per_device = int(2e3)

#         self.expected_train_count = self.num_windowed_examples_per_device * len(ALL_SERIAL_NUMBERS)
#         self.expected_val_count = self.num_val_examples_per_device * len(ALL_SERIAL_NUMBERS)
#         self.expected_test_count = self.num_test_examples_per_device * len(ALL_SERIAL_NUMBERS)

#         input_shuffled_ds_num_samples_per_chunk=4*ORIGINAL_PAPER_SAMPLES_PER_CHUNK
#         output_window_size=ORIGINAL_PAPER_SAMPLES_PER_CHUNK
#         stride_length=1

#         self.expected_train_replication_factor = math.ceil((input_shuffled_ds_num_samples_per_chunk - output_window_size)/stride_length + 1)
#         self.expected_val_replication_factor = math.ceil(input_shuffled_ds_num_samples_per_chunk/output_window_size)
#         self.expected_test_replication_factor = math.ceil(input_shuffled_ds_num_samples_per_chunk/output_window_size)

#         self.shuffler = Windowed_Dataset_Shuffler(
#             input_shuffled_ds_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/all_shuffled_chunk-512/output",
#             input_shuffled_ds_num_samples_per_chunk=input_shuffled_ds_num_samples_per_chunk,
#             output_batch_size=100,
#             seed=1337,
#             num_windowed_examples_per_device=int(200e3),
#             num_val_examples_per_device=int(10e3),
#             num_test_examples_per_device=int(50e3),
#             output_max_file_size_MB=100,
#             output_window_size=output_window_size, 
#             serials_to_filter_on=ALL_SERIAL_NUMBERS,
#             working_dir=SCRATCH_DIR,
#             output_format_str="batch-{batch_size}_part-{part}.tfrecord_ds",
#             stride_length=stride_length
#         )

#         # clear_scratch_dir()
#         # self.shuffler.create_and_check_dirs()
#         # print("Write piles")
#         # self.shuffler.write_piles()
#         # print("shuffle")
#         # self.shuffler.shuffle_piles()

#     @unittest.skip("Skip cardinality to save time")
#     def test_cardinality(self):
#         # There's some slop in this due to the replication factor. Err on the side of caution and make sure there are more than what we wanted
#         # but at a max of below percent
#         acceptable_cardinality_delta_percent = 0.05
        
#         datasets = Windowed_Shuffled_Dataset_Factory(SCRATCH_DIR)

#         train_ds = datasets["train_ds"]
#         val_ds = datasets["val_ds"]
#         test_ds = datasets["test_ds"]   

#         train_count = 0
#         for e in train_ds:
#             train_count += e["index_in_file"].shape[0]

#         val_count = 0
#         for e in val_ds:
#             val_count += e["index_in_file"].shape[0]

#         test_count = 0
#         for e in test_ds:
#             test_count += e["index_in_file"].shape[0]


#         self.assertAlmostEqual(
#             train_count, 
#             self.expected_train_count, 
#             delta=self.expected_train_count*acceptable_cardinality_delta_percent
#         )
#         self.assertAlmostEqual(
#             val_count,
#             self.expected_val_count,
#             delta=self.expected_val_count*acceptable_cardinality_delta_percent
#         )
#         self.assertAlmostEqual(
#             test_count,
#             self.expected_test_count,
#             delta=self.expected_test_count*acceptable_cardinality_delta_percent
#         )
        
#         self.assertGreaterEqual(
#             train_count, 
#             self.expected_train_count, 
#         )
#         self.assertGreaterEqual(
#             val_count,
#             self.expected_val_count,
#         )
#         self.assertGreaterEqual(
#             test_count,
#             self.expected_test_count,
#         )

        
#     # @unittest.skip("Skip checking duplicates to save time")
#     def test_for_duplicates(self):
#         import functools
#         acceptable_cardinality_delta_percent = 0.10

#         """Make sure no chunk ID is shared between the datasets"""
#         datasets = Windowed_Shuffled_Dataset_Factory(SCRATCH_DIR)

#         train_ds = datasets["train_ds"]
#         val_ds = datasets["val_ds"]
#         test_ds = datasets["test_ds"] 

#         all_ds = train_ds.concatenate(val_ds).concatenate(test_ds)

#         # train_ds = train_ds.unbatch().batch(100).prefetch(2).

#         for e in train_ds.take(1):
#             print(e)

#         print("reduce")
#         train_list = functools.reduce(np.concatenate, train_ds.as_numpy_iterator())


        
            
        
#         sys.exit(1)

#         train_hashes = []
#         for e in train_ds:
#             train_hashes.extend(
#                 list(e["index_in_file"].numpy())
#             )
        
#         val_hashes = []
#         for e in val_ds:
#             val_hashes.extend(
#                 list(e["index_in_file"].numpy())
#             )

#         test_hashes = []
#         for e in test_ds:
#             test_hashes.extend(
#                     list(e["index_in_file"].numpy())
#             )

#         all_hashes = []
#         for e in all_ds.unbatch():
#             all_hashes.extend(
#                 list(e["index_in_file"].numpy())
#             )


#         self.assertAlmostEqual(
#             len(val_hashes),
#             len(set(val_hashes)) * self.expected_val_replication_factor,
#             delta=len(set(val_hashes)) * self.expected_val_replication_factor * acceptable_cardinality_delta_percent
#         )

#         self.assertAlmostEqual(
#             len(test_hashes),
#             len(set(test_hashes)) * self.expected_test_replication_factor,
#             delta=len(set(test_hashes)) * self.expected_test_replication_factor * acceptable_cardinality_delta_percent
#         )

#         train_hashes = set(train_hashes)
#         val_hashes   = set(val_hashes)
#         test_hashes  = set(test_hashes)

#         self.assertEqual(
#             len(train_hashes.intersection([val_hashes])),
#             0
#         )

#         self.assertEqual(
#             len(train_hashes.intersection([test_hashes])),
#             0
#         )

#         self.assertEqual(
#             len(val_hashes.intersection([test_hashes])),
#             0
#         )

#         # self.assertTrue(len(test_hashes) == len(set(test_hashes)))

#         # self.assertTrue(
#         #     len(val_hashes+test_hashes) == len(set(val_hashes+test_hashes))
#         # )

#     def test_shuffling(self):
#         """
#         This one is a bit hard. How do you check for randomness?

#         What I ended up doing is taking the 'index_in_file' metadata field, sorting it, and comparing it to the original.
#         If they aren't the same then we should be good.
#         """

#         datasets = Windowed_Shuffled_Dataset_Factory(SCRATCH_DIR)

#         train_ds = datasets["train_ds"]

#         indices = []
#         for e in train_ds:
#             indices.extend(e["index_in_file"].numpy())
                    
#         sorted_indices = copy.deepcopy(indices)
#         sorted_indices.sort()

#         self.assertFalse(
#             np.array_equal(
#                 indices,
#                 sorted_indices
#             )
#         )



        

if __name__ == "__main__":
    unittest.main()



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