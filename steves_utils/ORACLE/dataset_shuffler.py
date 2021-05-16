#! /usr/bin/env python3

import steves_utils.ORACLE.serialization as oracle_serialization
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_DISTANCES_FEET, ALL_RUNS, ALL_SERIAL_NUMBERS
import steves_utils.dataset_shuffler
import tensorflow as tf

from typing import List

import os

class Dataset_Shuffler:
    """Shuffles tensorflow datasets on disk
    Workflow (See method docstring for details):
            shuffler.create_and_check_dirs()
                This is optional. Creates the directory structure (pile and output dir)
            shuffler.write_piles()
                Write the piles to the output dir
            shuffler.shuffle_piles()
                Shuffle the piles in memory, concatenate them to the output dir.

    Basics:
        This class uses TFRecords to facilitate shuffling large datasets on disk. This is
        done by dropping the input examples into random piles which are small enough to fit
        in memory. These piles are then shuffled, and the output is concatenated to split 
        output files.
    
    NOTE:
        Care should be taken that num_piles results in piles which are small enough to fit in memory,
        but also large enough that reading from them is efficient (several GB is appropriate).

    NOTE:
        partial batches CAN be generated
    """

    def __init__(
        self,
        output_batch_size,
        num_piles,
        output_max_file_size_MB,
        pile_dir,
        output_dir,
        seed,
        num_samples_per_chunk,
        distances_to_get: List[int] = ALL_DISTANCES_FEET, 
        serial_numbers_to_get: List[str] =ALL_SERIAL_NUMBERS,
        runs_to_get: List[int] = ALL_RUNS,
        output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
        fail_on_too_few_output_parts=True,
    ) -> None:
        self.output_batch_size       = output_batch_size
        self.num_piles               = num_piles
        self.output_max_file_size_MB = output_max_file_size_MB
        self.pile_dir                = pile_dir
        self.output_dir              = output_dir
        self.seed                    = seed
        self.num_samples_per_chunk   = num_samples_per_chunk
        self.distances_to_get        = distances_to_get
        self.serial_numbers_to_get   = serial_numbers_to_get
        self.runs_to_get             = runs_to_get
        self.output_format_str       = output_format_str

        

        self.ds, self.cardinality = Simple_ORACLE_Dataset_Factory(
            num_samples_per_chunk, 
            runs_to_get=runs_to_get,
            distances_to_get=distances_to_get,
            serial_numbers_to_get=serial_numbers_to_get
        )

        total_ds_size_GB = self.cardinality * self.num_samples_per_chunk * 8 * 2 / 1024 / 1024 / 1024
        expected_pile_size_GB = total_ds_size_GB / self.num_piles
        if expected_pile_size_GB > 5:
            raise Exception("Expected pile size is too big: {}GB. Increase your num_piles".format(expected_pile_size_GB))

        expected_num_parts = total_ds_size_GB * 1024 / output_max_file_size_MB
        if expected_num_parts < 15:
            if fail_on_too_few_output_parts:
                raise Exception("Expected number of output parts is {}, need a minimum of 15".format(expected_num_parts))
            else:
                print("Expected number of output parts is {}, need a minimum of 15".format(expected_num_parts))


        self.shuffler = steves_utils.dataset_shuffler.Dataset_Shuffler(
            input_ds=self.ds,
            one_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            one_example_from_serialized_tf_record_func=oracle_serialization.serialized_tf_record_to_example,
            batch_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            output_batch_size=output_batch_size,
            num_piles=num_piles,
            output_format_str=output_format_str,
            output_max_file_size_MB=output_max_file_size_MB,
            pile_dir=pile_dir,
            output_dir=output_dir,
            seed=seed
        )



    def create_and_check_dirs(self):
        self.shuffler.create_and_check_dirs()

    def shuffle_piles(self, reuse_piles=False):
        self.shuffler.shuffle_piles(reuse_piles)

    def write_piles(self):
        self.shuffler.write_piles()


if __name__ == "__main__":
    from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS
    shuffler = Dataset_Shuffler(
        num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
        output_batch_size=1000,
        num_piles=5,
        output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
        output_max_file_size_MB=200,
        # output_max_file_size_MB=1,
        pile_dir="/mnt/wd500GB/derp/pile",
        output_dir="/mnt/wd500GB/derp/output",
        seed=1337,
        runs_to_get=[1],
        distances_to_get=[8],
        serial_numbers_to_get=[ALL_SERIAL_NUMBERS[0]]
    )


    shuffler.create_and_check_dirs()
    print("Write piles")
    shuffler.write_piles()
    print("shuffle")
    shuffler.shuffle_piles()