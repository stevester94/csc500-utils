#! /usr/bin/python3
import io
import pprint
import re
import numpy as np
import os
import sys
import multiprocessing as mp
import random
import time
import tensorflow as tf

class FixedLengthDatasetAccessor():
    def __init__(
        self,
        seed,
        bin_path="/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/bin",
        day_to_get="All",
        transmitter_id_to_get="All",
        transmission_id_to_get="All",
    ):
        self.day_to_get = day_to_get
        self.transmitter_id_to_get = transmitter_id_to_get
        self.transmission_id_to_get = transmission_id_to_get
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

        self.paths = self.filter_datasets(self.get_binaries_in_dir(bin_path))

        self.paths = [
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-10_transmission-10.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-11_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-11_transmission-7.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-12_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-13_transmission-2.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-13_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-14_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-15_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-19_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-20_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-6_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-7_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-8_transmission-2.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-9_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-1_transmitter-9_transmission-9.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-10_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-11_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-11_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-14_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-14_transmission-6.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-15_transmission-10.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-15_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-16_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-17_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-18_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-18_transmission-7.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-19_transmission-10.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-1_transmission-2.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-3_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-5_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-7_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-8_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-2_transmitter-9_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-12_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-13_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-15_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-16_transmission-2.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-16_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-17_transmission-9.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-18_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-19_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-1_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-1_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-20_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-20_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-2_transmission-7.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-2_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-3_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-4_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-4_transmission-7.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-5_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-6_transmission-9.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-7_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-7_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-8_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-8_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-3_transmitter-9_transmission-2.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-10_transmission-10.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-11_transmission-10.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-11_transmission-2.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-11_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-11_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-14_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-14_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-16_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-1_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-20_transmission-7.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-2_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-2_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-4_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-4_transmission-7.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-4_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-5_transmission-10.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-5_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-5_transmission-7.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-6_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-6_transmission-6.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-8_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-8_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-4_transmitter-9_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-7_transmitter-14_transmission-6.bin",
            "../../csc500-dataset-preprocessor/bin/day-7_transmitter-14_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-7_transmitter-14_transmission-9.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-10_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-13_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-13_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-15_transmission-6.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-18_transmission-6.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-18_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-19_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-1_transmission-2.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-1_transmission-5.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-3_transmission-2.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-3_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-3_transmission-6.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-4_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-4_transmission-7.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-4_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-6_transmission-6.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-6_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-8_transmitter-9_transmission-7.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-10_transmission-1.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-10_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-10_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-11_transmission-6.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-11_transmission-9.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-12_transmission-2.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-14_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-16_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-16_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-18_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-18_transmission-7.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-1_transmission-3.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-2_transmission-10.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-4_transmission-4.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-4_transmission-8.bin",
            "../../csc500-dataset-preprocessor/bin/day-9_transmitter-7_transmission-1.bin",
        ]


        # self.rng.shuffle(self.paths)

        if len(self.paths) == 0:
            print("No paths remained after filtering. Time to freak out!")
            sys.exit(1)


        self.dataset = tf.data.FixedLengthRecordDataset(
            self.paths, record_bytes=384, header_bytes=None, footer_bytes=None, buffer_size=10*384,
            compression_type=None, num_parallel_reads=5
        )
        self.dataset = self.dataset.prefetch(1000)
        # self.dataset = self.dataset.batch(2000)

        # frequency_domain_IQ tf.float32 376 Bytes
        # For testing purposes we'll shave off 8 bytes (IE we're bastardizing 2 samples from the IQ portion to fake the metadata)
        # day tf.uint8
        # transmitter_id tf.uint8
        # transmission_id tf.uint8
        # symbol_index_in_file tf.uint32
        self.dataset = self.dataset.map(
           lambda x: (
                tf.strings.substr(x, 0, 376, unit='BYTE', name=None),
                tf.strings.substr(x, 376, 1, unit='BYTE', name=None),
                tf.strings.substr(x, 377, 1, unit='BYTE', name=None),
                tf.strings.substr(x, 378, 1, unit='BYTE', name=None),
                tf.strings.substr(x, 379, 4, unit='BYTE', name=None),
            ),
            num_parallel_calls=10
        )
        self.dataset = self.dataset.map(
           lambda frequency_domain_IQ,day,transmitter_id,transmission_id,symbol_index_in_file: (
                tf.io.decode_raw(frequency_domain_IQ, tf.float32),
                tf.io.decode_raw(day, tf.uint8),
                tf.io.decode_raw(transmitter_id, tf.uint8),
                tf.io.decode_raw(transmission_id, tf.uint8),
                tf.io.decode_raw(symbol_index_in_file, tf.int32),
            ),
            num_parallel_calls=10
        )

        self.dataset = self.dataset.map(
            lambda frequency_domain_IQ,day,transmitter_id,transmission_id,symbol_index_in_file: (
                tf.reshape(frequency_domain_IQ, (2,47)),
                day,
                transmitter_id,
                transmission_id,
                symbol_index_in_file,
            ),
            num_parallel_calls=10
        )

        # self.dataset = self.dataset.unbatch()
        self.dataset = self.dataset.batch(2000)

        print(self.dataset.element_spec)



    def get_binaries_in_dir(self, path):
        (_, _, filenames) = next(os.walk(path))
        return [os.path.join(path,f) for f in filenames if ".bin" in f]

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

def speed_test(iterable, batch_size=1):
    import time

    last_time = time.time()
    count = 0
    
    for i in iterable:
        count += 1

        if count % (10000/batch_size) == 0:
            items_per_sec = count / (time.time() - last_time)
            print("Items per second:", items_per_sec*batch_size)
            last_time = time.time()
            count = 0

if __name__ == "__main__":
    flda = FixedLengthDatasetAccessor(1337, bin_path="../../csc500-dataset-preprocessor/bin/",)

    ds = flda.dataset
    # ds = ds.take(1)
    # for e in ds:
    #     f = e

    speed_test(ds, batch_size=2000)
