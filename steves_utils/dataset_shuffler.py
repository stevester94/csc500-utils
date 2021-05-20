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
    """

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

        self.pile_writers = []

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
        for i in range(self.num_piles):
            path = os.path.join(self.pile_dir, "pile_{}".format(i))
            writer = tf.io.TFRecordWriter(path)
            self.pile_writers.append(writer)
        
    def _close_pile_writers(self):
        """Close all the writers once we are done"""
        for p in self.pile_writers:
            p.close()

    def shuffle_piles(self, reuse_piles=False):
        """For each file in the pile dir, open it and parse it using one_example_from_serialized_tf_record_func,
            shuffle it, and write the output to files. The output files are split based on output_max_file_size_MB
        """
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

            pile_in_ds = pile_in_ds.prefetch(100)

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
                current_file_size += len(serialized_tf_record)
                current_writer.write(serialized_tf_record)

        current_writer.close()

    def write_piles(self):
        """Each example read from the input_ds is written to a random pile, using one_example_to_tf_record_func
        """
        self._open_pile_writers()

        for e in self.input_ds.prefetch(100):
            tf_record = self.one_example_to_tf_record_func(e)
            serialized_tf_record = tf_record.SerializeToString()

            random_pile_writer_index = self.rng.integers(0, self.num_piles)
            random_pile_writer = self.pile_writers[random_pile_writer_index]

            random_pile_writer.write(serialized_tf_record)

        self._close_pile_writers()

    def __del__(self):
        for pw in self.pile_writers:
            pw.close()


def Shuffled_Dataset_Factory(
    path,
    train_val_test_splits,
    reshuffle_train_each_iteration=True
):
    """Generates TF datasets for accessing large shuffled datasets.
    train,val,test splits are done based on files rather than the commong take/skip method.

    Reshuffling of train_ds is done via reshuffling the file names on each iteration (I have confirmed this works)

    Yes I know it's not a factory.
    """
    files = utils.get_files_in_dir(path)

    train_start_stop = (0, int(len(files)*train_val_test_splits[0]))
    val_start_stop   = (train_start_stop[1], train_start_stop[1] +  int(len(files)*train_val_test_splits[1]))
    test_start_stop  = (val_start_stop[1], val_start_stop[1] + int(len(files)*train_val_test_splits[2]))

    train_files     = files[train_start_stop[0]:train_start_stop[1]]
    val_files       = files[val_start_stop[0]:val_start_stop[1]]
    test_files      = files[test_start_stop[0]:test_start_stop[1]]

    if len(train_files) == 0:
        raise Exception("Train files had no files, indicative of bad tran/val/test split or too few files to draw from")
    if len(val_files) == 0:
        raise Exception("Val files had no files, indicative of bad tran/val/test split or too few files to draw from")
    if len(test_files) == 0:
        raise Exception("Test files had no files, indicative of bad tran/val/test split or too few files to draw from")

    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    val_ds   = tf.data.Dataset.from_tensor_slices(val_files)
    test_ds  = tf.data.Dataset.from_tensor_slices(test_files)

    if reshuffle_train_each_iteration:
        train_ds = train_ds.shuffle(train_ds.cardinality(), reshuffle_each_iteration=True)

    # SM: OK let's unpack what's going on here. 
    # We start with 'dataset' which is a dataset of file paths.
    # We then use 'interleave to apply a map function across this dataset of file paths and then interleave the results
    # The map function itself is creating a TFRectordDataset from the path. Normally you would then map this using your custom TFRecord deserialize
    # function. This factory doesn't care about that, we leave it to the downstream user.
    #
    # Because the files are already shuffled, the "interleaving" only pulls from one file at a time (As determined by cycle length).
    #     There may be performance gains if this is increased.
    #
    # Explanation (Note I wrote these when I was using interleaving as a means of shuffling, so they are in that context, not the current usage):
    # block_length: How many elements we want to pull from each dataset before going to the next
    #               Note that this is a little wonky since each TFRecord path is a file that only contains a single TFRecord. However, this may
    #               not always be the case. So to achieve good shuffling block length should be 1 
    # cycle_length: Should be set to the number of datasets. This is how many original datasets we operate on at the same time.
    #               So if we had 10 datasets we are interleaving, and block length one, but cycle length = 5, we'd pull one element
    #               from each of those 5 datasets before moving to the next set of 5.

    train_ds = train_ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=1, 
        block_length=1,
        deterministic=True
    )
    val_ds = val_ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=1, 
        block_length=1,
        deterministic=True
    )
    test_ds = test_ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=1, 
        block_length=1,
        deterministic=True
    )

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
    }


if __name__ == "__main__":
    from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
    from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS
    import steves_utils.ORACLE.serialization as oracle_serialization

    ds, cardinality = Simple_ORACLE_Dataset_Factory(
        ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
        runs_to_get=[1],
        distances_to_get=[8],
        serial_numbers_to_get=[ALL_SERIAL_NUMBERS[0]]
    )

    # shuffler = Dataset_Shuffler(
    #     input_ds=ds,
    #     one_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
    #     one_example_from_serialized_tf_record_func=oracle_serialization.serialized_tf_record_to_example,
    #     batch_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
    #     output_batch_size=1000,
    #     num_piles=5,
    #     output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
    #     output_max_file_size_MB=200,
    #     pile_dir="/mnt/wd500GB/derp_2/pile",
    #     output_dir="/mnt/wd500GB/derp_2/output",
    #     seed=1337
    # )
    
    
    # p = "/mnt/wd500GB/derp/pile/pile_0"

    # for p in utils.get_files_in_dir("/mnt/wd500GB/derp/pile/"):
    #     ds = tf.data.TFRecordDataset(p).map(oracle_serialization.serialized_tf_record_to_example)

    #     for e in ds:
    #         print(e["index_in_file"])


    # shuffler.create_and_check_dirs()
    # shuffler.write_piles()
    # print("Now shuffle")
    # shuffler.shuffle_piles()

    datasets = Shuffled_Dataset_Factory("/mnt/wd500GB/derp/output", train_val_test_splits=(0.6, 0.2, 0.2))

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

    train_ds = train_ds.map(oracle_serialization.serialized_tf_record_to_example)

    count = 0
    for e in train_ds:
        count += e["IQ"].shape[0]
        print(count)