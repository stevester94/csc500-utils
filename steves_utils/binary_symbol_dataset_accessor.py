#! /usr/bin/python3
import io
import pprint
import re
import numpy as np
import os
import sys
import tensorflow as tf

from binary_random_accessor import Binary_OFDM_Symbol_Random_Accessor

pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint

def get_binaries_in_dir(path):
    (_, _, filenames) = next(os.walk(path))
    return [os.path.join(path,f) for f in filenames if ".bin" in f]

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

class BinarySymbolDatasetAccessor():
    def __init__(
        self,
        tfrecords_path="/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/bin",
        day_to_get="All",
        transmitter_id_to_get="All",
        transmission_id_to_get="All",
        parallelism=2
    ):
        self.day_to_get = day_to_get
        self.transmitter_id_to_get = transmitter_id_to_get
        self.transmission_id_to_get = transmission_id_to_get

        self.paths = self.filter_datasets(get_binaries_in_dir(tfrecords_path))

        if len(self.paths) == 0:
            print("No paths remained after filtering. Time to freak out!")
            sys.exit(1)
        
        print("########################################################")
        print("########################################################")
        print("TODO GET SEED")
        print("########################################################")
        print("########################################################")

        rng = np.random.default_rng(1337)
        rng.shuffle(self.paths)

        split_paths = split(self.paths, parallelism)

        datasets = []
        self.cardinality = 0
        for sp in split_paths:
            bosra = Binary_OFDM_Symbol_Random_Accessor(sp, 1337)
            self.cardinality += bosra.get_dataset_cardinality()
            datasets.append(self._ds_from_BOSRA(bosra))
        
        self.dataset = datasets[0]

        for ds in datasets[1:]:
            self.dataset.concatenate(ds)
        



        return

        dataset = tf.data.Dataset.from_tensor_slices(self.tfrecords_paths)
        dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=False)

        # SM: OK let's unpack what's going on here. 
        # We start with 'dataset' which is a dataset of file paths.
        # We then use 'interleave to apply a map function across this dataset of file paths and then interleave the results
        # The map function itself is creating a TFRectordDataset from the path, which itself is then calling a map function to deserialize the TFRecords
        # block_length: How many elements we want to pull from each dataset before going to the next
        #               Note that this is a little wonky since each TFRecord path is a file that only contains a single TFRecord. However, this may
        #               not always be the case. So to achieve good shuffling block length should be 1 
        # cycle_length: Should be set to the number of datasets. This is how many original datasets we operate on at the same time.
        #               So if we had 10 datasets we are interleaving, and block length one, but cycle length = 5, we'd pull one element
        #               from each of those 5 datasets before moving to the next set of 5.
        self.dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=dataset.cardinality(), 
            block_length=10,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )
        self.dataset = self.dataset.map(self.parse_serialized_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        self.dataset = self.dataset.prefetch(10000)

        self.dataset_cardinality = -1

    def _ds_from_BOSRA(self, bosra):
        ds = tf.data.Dataset.from_generator(
            bosra.generator,
            output_types={
                "transmitter_id": tf.int64,
                "day": tf.int64,
                "transmission_id": tf.int64,
                "frequency_domain_IQ": tf.float32,
                "frame_index": tf.int64,
                "symbol_index": tf.int64,
            },
            output_shapes={
                "transmitter_id": (),
                "day": (),
                "transmission_id": (),
                "frequency_domain_IQ": (2,48),
                "frame_index": (),
                "symbol_index": (),
            }
        )

        return ds

    def get_dataset(self):
        return self.dataset

    def is_any_word_in_string(self, list_of_words, string):
        for w in list_of_words:
            if w in string:
                return True
        return False

    def filter_datasets(self, paths):
        filtered_paths = paths
        if self.day_to_get != "All":
            assert(isinstance(self.day_to_get, list))
            assert(len(self.day_to_get) > 0)
            
            filt = ["day-{}_".format(f) for f in self.day_to_get]
            filtered_paths = [p for p in filtered_paths if self.is_any_word_in_string(filt, p)]

        if self.transmitter_id_to_get != "All":
            assert(isinstance(self.transmitter_id_to_get, list))
            assert(len(self.transmitter_id_to_get) > 0)

            filt = ["transmitter-{}_".format(f) for f in self.transmitter_id_to_get]
            filtered_paths = [p for p in filtered_paths if self.is_any_word_in_string(filt, p)]

        if self.transmission_id_to_get != "All":
            assert(isinstance(self.transmission_id_to_get, list))
            assert(len(self.transmission_id_to_get) > 0)

            filt = ["transmission-{}.".format(f) for f in self.transmission_id_to_get]
            filtered_paths = [p for p in filtered_paths if self.is_any_word_in_string(filt, p)]

        return filtered_paths


    def get_paths(self):
        return self.paths
    
    def get_dataset_cardinality(self):
        return self.cardinality

if __name__ == "__main__":
    bsda = BinarySymbolDatasetAccessor(day_to_get=[1])

    ds = bsda.get_dataset()

    for e in ds:
        print(e)