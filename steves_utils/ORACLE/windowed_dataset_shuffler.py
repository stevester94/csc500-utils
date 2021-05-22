#! /usr/bin/env python3

import steves_utils.ORACLE.serialization as oracle_serialization
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_DISTANCES_FEET, ALL_RUNS, ALL_SERIAL_NUMBERS, serial_number_to_id
from steves_utils.ORACLE.shuffled_dataset_accessor import Shuffled_Dataset_Factory
import steves_utils.dataset_shuffler
import tensorflow as tf
import steves_utils.utils as utils

from functools import reduce

from typing import List
import math

import os

class Windowed_Dataset_Shuffler:
    """Alright this one is a real doozy
    Will take an already shuffled and chunked dataset (basically use all_shuffled_chunk-512) and
        output the following:
        - A windowed and shuffled train dataset
        - A monolithic validation dataset file (not windowed)
        - A monolithic test dataset file (not windowed)
    
    The total desired number of examples _PER DEVICE_ for the train/test/val datasets are specified.
    
    The train/test/val datasets are guaranteed to not contain examples that are related to each other 
        at all (IE they will not share chunks at _all_)

    

    Workflow (See method docstring for details):
            shuffler.create_and_check_dirs()
                This is optional. Creates the directory structure (pile and output dir)
            shuffler.write_piles()
                Write the piles to the output dir
            shuffler.shuffle_piles()
                Shuffle the piles in memory, concatenate them to the output dir.
    """

    def __init__(
        self,
        input_shuffled_ds_dir,
        input_shuffled_ds_num_samples_per_chunk,
        output_batch_size,
        output_max_file_size_MB,
        seed,
        num_windowed_examples_per_device,
        num_val_examples_per_device,
        num_test_examples_per_device,
        output_window_size,
        serials_to_filter_on,
        window_pile_dir,
        window_output_dir,
        window_output_format_str,
        val_pile_dir,
        val_output_dir,
        val_output_format_str,
        test_pile_dir,
        test_output_dir,
        test_output_format_str,
        stride_length,
        # output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
    ) -> None:
        self.serial_ids_to_filter_on                 = [serial_number_to_id(serial) for serial in  serials_to_filter_on]
        self.num_windowed_examples_per_device        = num_windowed_examples_per_device
        self.num_val_examples_per_device             = num_val_examples_per_device
        self.num_test_examples_per_device            = num_test_examples_per_device
        self.window_pile_dir                         = window_pile_dir
        self.window_output_dir                       = window_output_dir
        self.window_output_format_str                = window_output_format_str
        self.val_pile_dir                            = val_pile_dir
        self.val_output_dir                          = val_output_dir
        self.val_output_format_str                   = val_output_format_str
        self.test_pile_dir                           = test_pile_dir
        self.test_output_dir                         = test_output_dir
        self.test_output_format_str                  = test_output_format_str
        self.input_shuffled_ds_num_samples_per_chunk = input_shuffled_ds_num_samples_per_chunk
        self.output_batch_size                       = output_batch_size
        self.output_max_file_size_MB                 = output_max_file_size_MB
        self.seed                                    = seed
        self.output_window_size                      = output_window_size

        self.num_devices = len(self.serial_ids_to_filter_on)
        
        # Yeah it's pretty hacky since we don't really need to split the dataset into test and val, but
        # it's already written and tested
        datasets = Shuffled_Dataset_Factory(
            input_shuffled_ds_dir, train_val_test_splits=(0.6, 0.2, 0.2), reshuffle_train_each_iteration=False
        )

        self.train_ds = datasets["train_ds"].unbatch()
        self.val_ds = datasets["val_ds"].unbatch()
        self.test_ds = datasets["test_ds"].unbatch()

        # Since we are windowing, the number of examples we take from the original dataset is smaller
        # than the actual number of windows we want to generate
        replication_factor = math.floor((input_shuffled_ds_num_samples_per_chunk - output_window_size)/stride_length + 1)
        num_train_examples_to_get_per_device = num_windowed_examples_per_device/replication_factor

        self.train_ds = Windowed_Dataset_Shuffler.build_per_device_filtered_dataset(
            serial_ids_to_filter_on=self.serial_ids_to_filter_on,
            num_examples_per_serial_id=num_train_examples_to_get_per_device, # THIS IS WRONG
            ds=self.train_ds,
        )
        self.val_ds = Windowed_Dataset_Shuffler.build_per_device_filtered_dataset(
            serial_ids_to_filter_on=self.serial_ids_to_filter_on,
            num_examples_per_serial_id=num_val_examples_per_device,
            ds=self.train_ds,
        )
        self.test_ds = Windowed_Dataset_Shuffler.build_per_device_filtered_dataset(
            serial_ids_to_filter_on=self.serial_ids_to_filter_on,
            num_examples_per_serial_id=num_test_examples_per_device,
            ds=self.train_ds,
        )

        self.train_ds = self.window_ds(self.train_ds)

        # This is another straight up hack. The val and test aren't really shuffled, we're just using this to write the DS to file
        self.train_shuffler = self.make_train_shuffler()
        self.val_shuffler   = self.make_val_shuffler()
        self.test_shuffler  = self.make_test_shuffler()



    @staticmethod
    def build_per_device_filtered_dataset(
        serial_ids_to_filter_on,
        num_examples_per_serial_id,
        ds,
    ):
        """Filters and takes the appropriate number of examples from each device. Does not do any mapping/windowing"""

        datasets = []

        for serial in serial_ids_to_filter_on:
            datasets.append(ds.filter(lambda x: x["serial_number_id"] == serial).take(num_examples))

        return reduce(lambda a,b: a.concatenate(b), datasets)

    def window_ds(self, ds):
        """Applies our window function across the already shuffled dataset"""        

        num_repeats= math.floor((self.input_shuffled_ds_num_samples_per_chunk - self.output_window_size)/self.stride_length) + 1

        ds = ds.map(
            lambda x: {
                "IQ": tf.transpose(
                    tf.signal.frame(x["IQ"], self.output_window_size, self.stride_length),
                    [1,0,2]
                ),
                "index_in_file": tf.repeat(tf.reshape(x["index_in_file"], (1, x["index_in_file"].shape[0])), repeats=num_repeats, axis=0),
                "serial_number_id": tf.repeat(tf.reshape(x["serial_number_id"], (1, x["serial_number_id"].shape[0])), repeats=num_repeats, axis=0),
                "distance_feet": tf.repeat(tf.reshape(x["distance_feet"], (1, x["distance_feet"].shape[0])), repeats=num_repeats, axis=0),
                "run": tf.repeat(tf.reshape(x["run"], (1, x["run"].shape[0])), repeats=num_repeats, axis=0),
            },
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

        ds = ds.unbatch()

        return ds

    def make_train_shuffler(self) -> steves_utils.dataset_shuffler.Dataset_Shuffler:
        ds_size_GB =  self.num_windowed_examples_per_device * self.num_devices * self.output_window_size * 8 * 2 / 1024 / 1024 / 1024
        num_piles = int(math.ceil(ds_size_GB))

        self.expected_num_parts = ds_size_GB * 1024 / self.output_max_file_size_MB
        if self.expected_num_parts < 15:
            raise Exception("Expected number of output parts is {}, need a minimum of 15".format(self.expected_num_parts))


        return steves_utils.dataset_shuffler.Dataset_Shuffler(
            input_ds=self.train_ds,
            one_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            one_example_from_serialized_tf_record_func=oracle_serialization.serialized_tf_record_to_example,
            batch_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            output_batch_size=self.output_batch_size,
            num_piles=num_piles,
            output_format_str=self.window_output_format_str,
            output_max_file_size_MB=self.output_max_file_size_MB,
            pile_dir=self.window_pile_dir,
            output_dir=self.window_output_dir,
            seed=self.seed
        )


    def make_val_shuffler(self) -> steves_utils.dataset_shuffler.Dataset_Shuffler:
        return steves_utils.dataset_shuffler.Dataset_Shuffler(
            input_ds=self.val_ds,
            one_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            one_example_from_serialized_tf_record_func=oracle_serialization.serialized_tf_record_to_example,
            batch_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            output_batch_size=self.output_batch_size,
            num_piles=1,
            output_format_str=self.val_output_format_str,
            output_max_file_size_MB=100*1024,
            pile_dir=self.val_pile_dir,
            output_dir=self.val_output_format_str,
            seed=self.seed,
        )


    def make_test_shuffler(self) -> steves_utils.dataset_shuffler.Dataset_Shuffler:
        return steves_utils.dataset_shuffler.Dataset_Shuffler(
            input_ds=self.test_ds,
            one_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            one_example_from_serialized_tf_record_func=oracle_serialization.serialized_tf_record_to_example,
            batch_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            output_batch_size=self.output_batch_size,
            num_piles=1,
            output_format_str=self.test_output_format_str,
            output_max_file_size_MB=100*1024,
            pile_dir=self.test_pile_dir,
            output_dir=self.test_output_dir,
            seed=self.seed,
        )

    def get_total_ds_size_GB(self):
        return self.total_ds_size_GB
    def get_expected_pile_size_GB(self):
        return self.expected_pile_size_GB
    def get_expected_num_parts(self):
        return self.expected_num_parts

    def create_and_check_dirs(self):
        for s in (self.train_shuffler, self.val_shuffler,self.test_shuffler):
            s.create_and_check_dirs()

    def shuffle_piles(self, reuse_piles=False):
        for s in (self.train_shuffler, self.val_shuffler,self.test_shuffler):
            s.shuffle_piles(reuse_piles)

    def write_piles(self):
        for s in (self.train_shuffler, self.val_shuffler,self.test_shuffler):
            s.write_piles()
    
    def get_num_piles(self):
        return self.num_piles


if __name__ == "__main__":
    from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS
    shuffler = Windowed_Dataset_Shuffler(
        input_shuffled_ds_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/all_shuffled_chunk-512/output",
        input_shuffled_ds_num_samples_per_chunk=4*ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
        output_batch_size=100,
        output_max_file_size_MB=100,
        seed=1337,
        num_windowed_examples_per_device=int(200e3),
        num_val_examples_per_device=int(10e3),
        num_test_examples_per_device=int(50e3),
        output_window_size=ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
        serials_to_filter_on=ALL_SERIAL_NUMBERS,
        window_pile_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/windowed_200k-each-devices_batch-100/window_pile",
        window_output_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/windowed_200k-each-devices_batch-100/window_output",
        window_output_format_str="shuffled_window-128_batchSize-{batch_size}_part-{part}.tfrecord_ds",
        val_pile_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/windowed_200k-each-devices_batch-100/val_pile",
        val_output_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/windowed_200k-each-devices_batch-100/val_output",
        val_output_format_str="validation_batch-{batch_size}_part-{part}.tfrecord_ds",
        test_pile_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/windowed_200k-each-devices_batch-100/test_pile",
        test_output_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/windowed_200k-each-devices_batch-100/test_output",
        test_output_format_str="test_batch-{batch_size}_part-{part}.tfrecord_ds"
    )


    shuffler.create_and_check_dirs()

    print("Num piles:", shuffler.get_num_piles())
    # input("Press enter to continue")
    # print("Write piles")
    # shuffler.write_piles()
    # print("shuffle")
    # shuffler.shuffle_piles()