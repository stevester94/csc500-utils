#! /usr/bin/python3

import os
import sys
import tensorflow as tf
from abc import ABC, abstractmethod
import re

import pprint
pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint



# Will only get files in the path, will not recurse to sub-dirs
def get_tfrecords_in_dir(path):
    (_, _, filenames) = next(os.walk(path))
    return [os.path.join(path,f) for f in filenames if ".tfrecord" in f]

class DatasetAccessor(ABC):
    def __init__(
        self,
        tfrecords_path="../vanilla_tfrecords/"
    ):
        self.tfrecords_paths = self.filter_datasets(get_tfrecords_in_dir(tfrecords_path))

        if len(self.tfrecords_paths) == 0:
            print("No paths remained after filtering. Time to freak out!")
            sys.exit(1)
        
        dataset = tf.data.Dataset.from_tensor_slices(self.tfrecords_paths)

        self.dataset = dataset.interleave(lambda path:
           tf.data.TFRecordDataset(path).map(self.parse_serialized_example, num_parallel_calls=1),
            cycle_length=4, block_length=16)

    def get_dataset(self):
        return self.dataset

    @abstractmethod 
    def filter_datasets(self, paths):
        pass

    @abstractmethod 
    def parse_serialized_example(self, example):
        pass

    def get_tfrecords_paths(self):
        return self.tfrecords_paths

class VanillaDatasetAccessor(DatasetAccessor):
    def __init__(
        self,
        day_to_get="All",
        transmitter_id_to_get="All",
        transmission_id_to_get="All"
    ):
        self.day_to_get = day_to_get
        self.transmitter_id_to_get = transmitter_id_to_get
        self.transmission_id_to_get = transmission_id_to_get

        # Contantly re-instantiating this dict adds non-trivial overhead, so we save it
        self._transmission_example_description = {
            'transmitter_id':     tf.io.FixedLenFeature([], tf.int64, default_value=-1337),
            'day':                tf.io.FixedLenFeature([], tf.int64, default_value=-1337),
            'transmission_id':    tf.io.FixedLenFeature([], tf.int64, default_value=-1337),
            'time_domain_IQ':    tf.io.FixedLenFeature([], tf.string, default_value='RIP'),
            'sha512_of_original': tf.io.FixedLenFeature([], tf.string, default_value='DEAD'),
        }

        super().__init__()

    def parse_serialized_transmission_example(self, serialized_example):
        parsed_example = tf.io.parse_single_example(serialized_example, self._transmission_example_description)

        parsed_example["time_domain_IQ"] = tf.io.parse_tensor(parsed_example["time_domain_IQ"], tf.float32)

        # Note that you can actually do some pretty tricky shit here such as
        #return parsed_example["time_domain_IQ"], parsed_example["device_id"]
        #return parsed_example["device_id"]

        return parsed_example

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

    def parse_serialized_example(self, example):
        return self.parse_serialized_transmission_example(example)


def unit_test():
    import random
    random.seed(1337)

    all_transmitters =           list(range(1,21))
    all_days =           list(range(1,10))
    all_transmissions =           list(range(1,11))

    def check_everything_is_there(days, transmitters, transmissions):
        dsa = VanillaDatasetAccessor(day_to_get=days, transmitter_id_to_get=transmitters, transmission_id_to_get=transmissions)
        ds = dsa.get_dataset()

        found_days=[]
        found_transmitters=[]
        found_transmissions=[]

        for i in ds:
            found_days.append(i["day"].numpy())
            found_transmitters.append(i["transmitter_id"].numpy())
            found_transmissions.append(i["transmission_id"].numpy())
        
        if days != "All":
            assert(set(found_days) == set(days))
        else:
            assert(set(found_days) == set(all_days))
        
        if transmitters != "All":
            assert(set(found_transmitters) == set(transmitters))
        else:
            assert(set(found_transmitters) == set(all_transmitters))

        # I'm not even gonna bother with the all case here, it's so fucked that if I really want to check for all I'll just specify them manually
        if transmissions != "All":
            assert(set(found_transmissions) == set(transmissions))

    print("Begin test 1")
    check_everything_is_there([3], [1], [1])
    print("Pass")

    print("Begin test 2")
    check_everything_is_there([2,3,4], "All", [1,2,3,4,5,6,7,8,9,10])
    print("Pass")

    print("Begin all test")
    check_everything_is_there(all_days, all_transmitters, all_transmissions)
    print("Pass")






    
if __name__ == "__main__":
    unit_test()
    print("Unit test passed")