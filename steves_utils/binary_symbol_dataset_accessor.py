#! /usr/bin/python3
import io
import pprint
import re
import numpy as np
import os
import sys
import tensorflow as tf

from steves_utils.binary_random_accessor import Binary_OFDM_Symbol_Random_Accessor

pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint

def get_binaries_in_dir(path):
    (_, _, filenames) = next(os.walk(path))
    return [os.path.join(path,f) for f in filenames if ".bin" in f]

def split(a, n):
    k, m = divmod(len(a), n)
    l = (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    l = list(l)
    return [thing for thing in l if len(thing) > 1]

class BinarySymbolDatasetAccessor():
    def __init__(
        self,
        seed,
        tfrecords_path="/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/bin",
        day_to_get="All",
        transmitter_id_to_get="All",
        transmission_id_to_get="All",
        parallelism=6
    ):
        self.day_to_get = day_to_get
        self.transmitter_id_to_get = transmitter_id_to_get
        self.transmission_id_to_get = transmission_id_to_get
        self.seed = seed

        self.paths = self.filter_datasets(get_binaries_in_dir(tfrecords_path))

        if len(self.paths) == 0:
            print("No paths remained after filtering. Time to freak out!")
            sys.exit(1)
        

        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.paths)

        split_paths = np.array_split(self.paths, min((len(self.paths), parallelism)))
        # split_paths = [str(path[0]) for path in split_paths]

        pprint(self.paths)

        print("Split paths")
        pprint(split_paths)

        datasets = [self._ds_from_paths(paths) for paths in split_paths]

        dataset = tf.data.Dataset.from_tensor_slices(datasets)
        # dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=False)
        # dataset = dataset.window(np.ceil( len(self.paths) / parallelism))


        dataset = dataset.interleave(
            # lambda x: self._ds_from_paths(list(x)),
            # lambda x: self._ds_from_paths(x),
            lambda x: x,
            cycle_length=dataset.cardinality(), 
            block_length=1,
            num_parallel_calls=min((len(self.paths), parallelism)),
            deterministic=True
        )
        
        dataset = dataset.prefetch(10000)
        self.dataset = dataset


        # Quick hack to get cardinality
        self.cardinality = Binary_OFDM_Symbol_Random_Accessor(self.paths, self.seed).get_dataset_cardinality()

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

    def _ds_from_paths(self, paths):
            return self._ds_from_BOSRA(Binary_OFDM_Symbol_Random_Accessor(paths, self.seed))

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