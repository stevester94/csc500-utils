#! /usr/bin/python3
import io
import pprint
import re
import numpy as np
import os
import sys
import tensorflow as tf
import multiprocessing as mp

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
        train_eval_test_splits=(0.6, 0.2, 0.2),
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
        self.train_eval_test_splits = train_eval_test_splits
        self.paths = self.filter_datasets(get_binaries_in_dir(bin_path))

        if len(self.paths) == 0:
            print("No paths remained after filtering. Time to freak out!")
            sys.exit(1)
        
        self.bosra = Binary_OFDM_Symbol_Random_Accessor(self.paths)

#####################################################################################

        self.cardinality = self.bosra.get_cardinality()

        # We generate our own indices based on the seed
        rng = np.random.default_rng(seed)
        indices = np.arange(0, self.cardinality)
        rng.shuffle(indices)

        # Build the train/eval/test indices lists
        train_size = int(self.cardinality * train_eval_test_splits[0])
        eval_size  = int(self.cardinality * train_eval_test_splits[1])
        test_size  = int(self.cardinality * train_eval_test_splits[2])

        self.train_indices = indices[:train_size]
        self.eval_indices  = indices[train_size:train_size+eval_size]
        self.test_indices  = indices[train_size+eval_size:]

        self.pool = mp.Pool(processes=8)


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
    
############################################

    def get_total_dataset_cardinality(self):
        return self.cardinality
        
    def get_train_dataset_cardinality(self):
        return len(self.train_indices)

    def get_eval_dataset_cardinality(self):
        return len(self.eval_indices)

    def get_test_dataset_cardinality(self):
        return len(self.test_indices)

    def train_generator(self):
        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.train_indices)

        for index in self.train_indices:
            yield self.bosra[index]

    def eval_generator(self):
        for index in self.eval_indices:
            yield self.bosra[index]

    def test_generator(self):
        for index in self.test_indices:
            yield self.bosra[index]

    def batch_generator_from_generator(self, generator_func, batch_size, repeat=False):
        gen = generator_func()
        while True:
            x = []
            y = []
            for i in range(batch_size):
                if repeat:
                    try:
                        e = next(gen)
                    except StopIteration:
                        gen = generator_func()
                        e = next(gen)
                else:
                    e = next(gen)
                x.append( e["frequency_domain_IQ"] )
                y.append( e["transmitter_id"] )

            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.int64)

            yield (x,y)

if __name__ == "__main__":
    RANGE   = 12
    BATCH   = 100
    EPOCHS  = 5
    DROPOUT = 0.5 # [0,1], the chance to drop an input

    bsda = BinarySymbolDatasetAccessor(
        seed=1337,
        batch_size=BATCH,
        num_class_labels=RANGE,
        bin_path="../../csc500-dataset-preprocessor/bin/",
        # day_to_get=[1],
        # transmitter_id_to_get=[10,11],
        transmission_id_to_get=[1],
    )

    # gen = bsda.test_generator(repeat=False)
    gen = bsda.test_generator()

    count = 0
    for i in gen:
        count += 1
    
    print("Total count:", count)