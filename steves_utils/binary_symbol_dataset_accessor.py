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


def _one_hot_encoder_generator(bosra_generator, num_class_labels):
    for e in bosra_generator:
        
        yield (
            e[0],
            tf.one_hot(tf.convert_to_tensor(e[1], dtype=tf.int64), num_class_labels) # One hot is quite slow, should be called on the top level tens
        )

class BinarySymbolDatasetAccessor():
    def __init__(
        self,
        seed,
        batch_size,
        num_class_labels,
        bin_path="/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/bin",
        day_to_get="All",
        transmitter_id_to_get="All",
        transmission_id_to_get="All",
    ):
        self.day_to_get = day_to_get
        self.transmitter_id_to_get = transmitter_id_to_get
        self.transmission_id_to_get = transmission_id_to_get
        self.seed = seed
        self.batch_size = batch_size
        self.num_class_labels = num_class_labels

        self.paths = self.filter_datasets(get_binaries_in_dir(bin_path))

        if len(self.paths) == 0:
            print("No paths remained after filtering. Time to freak out!")
            sys.exit(1)
        
        self.bosra = Binary_OFDM_Symbol_Random_Accessor(self.paths, self.seed)


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
    
    def get_total_dataset_cardinality(self):
        return self.bosra.get_total_dataset_cardinality()
        
    def get_train_dataset_cardinality(self):
        return self.bosra.get_train_dataset_cardinality()

    def get_eval_dataset_cardinality(self):
        return self.bosra.get_eval_dataset_cardinality()

    def get_test_dataset_cardinality(self):
        return self.bosra.get_test_dataset_cardinality()

    def get_train_generator(self):
        return _one_hot_encoder_generator(
            self.bosra.batch_generator_from_generator(self.bosra.train_generator, self.batch_size, repeat=True),
            self.num_class_labels
        )

    def get_eval_generator(self):
        return _one_hot_encoder_generator(
            self.bosra.batch_generator_from_generator(self.bosra.eval_generator, self.batch_size, repeat=True),
            self.num_class_labels
        )
        
    def get_test_generator(self):
        return _one_hot_encoder_generator(
            self.bosra.batch_generator_from_generator(self.bosra.test_generator, self.batch_size),
            self.num_class_labels
        )

if __name__ == "__main__":
    bsda = BinarySymbolDatasetAccessor(day_to_get=[1])

    ds = bsda.get_dataset()

    for e in ds:
        print(e)