#! /usr/bin/env python3

from steves_utils.ORACLE.serialization import serialized_tf_record_to_example
from tensorflow.python.data import util
from tensorflow.python.lib.io import tf_record
from steves_utils import utils
import tensorflow as tf
import numpy as np
import tensorflow as tf

import os

class Dataset_Shuffler:
    """Shuffles tensorflow datasets on disk"""

    def __init__(
        self,
        input_ds,
        one_example_to_tf_record_func,
        one_example_from_serialized_tf_record_func,
        batch_example_to_tf_record_func,
        output_batch_size,
        num_piles,
        output_max_file_size_MB,
        pile_dir,
        output_dir,
        seed,
        output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
    ) -> None:
        self.input_ds = input_ds
        self.one_example_to_tf_record_func = one_example_to_tf_record_func
        self.one_example_from_serialized_tf_record_func = one_example_from_serialized_tf_record_func
        self.batch_example_to_tf_record_func = batch_example_to_tf_record_func
        self.output_batch_size = output_batch_size
        self.num_piles = num_piles
        self.output_format_str = output_format_str
        self.output_max_file_size_MB = output_max_file_size_MB
        self.pile_dir = pile_dir
        self.output_dir = output_dir

        self.rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)

    def create_and_check_dirs(self):
        """Make sure the dirs exist and are empty"""

        if os.path.isfile(self.pile_dir):
            raise Exception("Requested pile dir is a file")
        if os.path.isfile(self.output_dir):
            raise Exception("Requested output dir is a file")

        if not os.path.isdir(self.pile_dir):
            os.mkdir(self.pile_dir)

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)


        if len(utils.get_files_in_dir(self.pile_dir)) != 0:
            raise Exception("Pile dir is not empty")
        if len(utils.get_files_in_dir(self.output_dir)) != 0:
            raise Exception("Out dir is not empty")

    def _open_pile_writers(self):
        """Open a TFRecordWriter for each pile"""
        self.pile_writers = []
        for i in range(self.num_piles):
            path = os.path.join(self.pile_dir, "pile_{}".format(i))
            writer = tf.io.TFRecordWriter(path)
            self.pile_writers.append(writer)

    # TODO: This doesn't need to be a separate step, we can dump it straight to the outputs
    def shuffle_piles(self, reuse_piles=False):
        if reuse_piles:
            raise Exception("Not Implemented")

        piles = utils.get_files_in_dir(self.pile_dir)

        current_part_count = 0
        current_file_size = 0
        current_writer = tf.io.TFRecordWriter(
            os.path.join(
                self.output_dir, self.output_format_str.format(batch_size=self.output_batch_size, part=current_part_count)
            )
        )

        for p in piles:
            pile_in_ds = tf.data.TFRecordDataset(p).map(self.one_example_from_serialized_tf_record_func)
            pile_in_ds = pile_in_ds.shuffle(100000000)

            if self.output_batch_size > 1:
                pile_in_ds = pile_in_ds.batch(self.output_batch_size)

            for e in pile_in_ds:
                tf_record = self.one_example_to_tf_record_func(e)
                serialized_tf_record = tf_record.SerializeToString()

                if current_file_size + len(serialized_tf_record) > self.output_max_file_size_MB * 1024 * 1024:
                    current_writer.close()
                    current_part_count += 1
                    current_file_size = 0
                    current_writer = tf.io.TFRecordWriter(
                        os.path.join(
                            self.output_dir, self.output_format_str.format(batch_size=self.output_batch_size, part=current_part_count)
                        )
                    )
                current_writer.write(serialized_tf_record)

    
    def write_piles(self):
        self._open_pile_writers()

        for e in self.input_ds:
            tf_record = self.one_example_to_tf_record_func(e)
            serialized_tf_record = tf_record.SerializeToString()

            random_pile_writer_index = self.rng.integers(0, self.num_piles)
            random_pile_writer = self.pile_writers[random_pile_writer_index]

            random_pile_writer.write(serialized_tf_record)

    def __del__(self):
        for pw in self.pile_writers:
            pw.close()



if __name__ == "__main__":
    from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
    from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS
    import steves_utils.ORACLE.serialization as oracle_serialization

    ds, cardinality = Simple_ORACLE_Dataset_Factory(
        ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
        # runs_to_get=[1],
        # distances_to_get=[8],
        # serial_numbers_to_get=[ALL_SERIAL_NUMBERS[0]]
    )

    shuffler = Dataset_Shuffler(
        input_ds=ds,
        one_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
        one_example_from_serialized_tf_record_func=oracle_serialization.serialized_tf_record_to_example,
        batch_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
        output_batch_size=1000,
        num_piles=500,
        output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
        output_max_file_size_MB=200,
        pile_dir="/mnt/wd500GB/derp/pile",
        output_dir="/mnt/wd500GB/derp/output",
        seed=1337
    )
    
    
    # p = "/mnt/wd500GB/derp/pile/pile_0"

    # for p in utils.get_files_in_dir("/mnt/wd500GB/derp/pile/"):
    #     ds = tf.data.TFRecordDataset(p).map(oracle_serialization.serialized_tf_record_to_example)

    #     for e in ds:
    #         print(e["index_in_file"])


    shuffler.create_and_check_dirs()
    shuffler.write_piles()
    print("Now shuffle")
    shuffler.shuffle_piles()