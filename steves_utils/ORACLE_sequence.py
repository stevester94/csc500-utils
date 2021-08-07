#! /usr/bin/env python3

import itertools
import numpy as np

import steves_utils.utils_v2 as steves_utils_v2

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_RUNS,
    ALL_SERIAL_NUMBERS,
    filter_paths,
    get_oracle_dataset_path,
    get_oracle_data_files_based_on_criteria
)

from steves_utils import file_as_windowed_list
from steves_utils import lazy_map
from steves_utils import sequence_aggregator
from steves_utils import sequence_mask
from steves_utils import sequence_cache

class ORACLE_Sequence:
    def __init__(
        self,
        desired_serial_numbers,
        desired_runs,
        desired_distances,
        window_length,
        window_stride,
        num_examples_per_device,
        seed,
        max_cache_size=1e6 # IDK
    ) -> None:
        self.rng = np.random.default_rng(seed)

        # We're going to group based on device so that we can aggregate them and then pull 'num_examples_per_device' from the aggregate.
        devices = {}
        for serial in desired_serial_numbers:
            devices[serial] = []
        
        """
        Get the data file path
        windowize the data file
        attach metadata to that windowed view of the datafile
        Group the metadata'd windowed view by device serial number
        """
        for serial_number, run, distance in set(itertools.product(desired_serial_numbers, desired_runs, desired_distances)):
            path = ORACLE_Sequence._get_a_data_file_path(serial_number, run, distance)
            windowed_sequence = ORACLE_Sequence._windowize_data_file(path, window_length, window_stride)
            windowed_sequence_with_metadata = ORACLE_Sequence._apply_metadata(windowed_sequence, serial_number,run,distance)

            devices[serial_number].append(windowed_sequence_with_metadata)
        

        """
        Aggregate them based on device
        Randomized_List_Mask them based on desired number of windows per device
        """
        masked_devices = []
        for device in devices.values():
            aggregated_device_sequence = sequence_aggregator.Sequence_Aggregator(device)
            mask = self.rng.integers(0, len(aggregated_device_sequence), num_examples_per_device) 
            masked_device = sequence_mask.Sequence_Mask(aggregated_device_sequence, mask)
            masked_devices.append(masked_device)
        

        """
        Final step!
        We aggregate all the devices
        Randomize across the entire aggregate (We use every example)
        Wrap this in a sequence cache
        """
        aggregated_devices = sequence_aggregator.Sequence_Aggregator(masked_devices)
        mask = self.rng.integers(0, len(aggregated_devices), len(aggregated_devices))
        masked_devices = sequence_mask.Sequence_Mask(aggregated_devices, mask)
        self.cache = sequence_cache.Sequence_Cache(masked_devices, max_cache_size)
    
    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]

    # The iterable could probably just be iter(self.cache)
    def __iter__(self):
        self.iter_idx = 0
        return self
    
    def __next__(self):
        if self.iter_idx >= len(self):
            raise StopIteration
        else:
            result = self[self.iter_idx]
            self.iter_idx += 1
            return result
    
    @staticmethod
    def _get_a_data_file_path(desired_serial_number, desired_run, desired_distance):
        paths = get_oracle_data_files_based_on_criteria(
            desired_serial_numbers=[desired_serial_number],
            desired_runs=[desired_run],
            desired_distances=[desired_distance],
        )
        assert len(paths) == 1
        path = paths[0]

        return path
    
    @staticmethod
    def _windowize_data_file(path, window_length, window_stride):
        faws = file_as_windowed_list.File_As_Windowed_Sequence(
            path=path,
            window_length=window_length,
            stride=window_stride,
            numpy_dtype=np.single
        )

        return faws
    
    @staticmethod
    def _apply_metadata(sequence, serial_number, run, distance):
        lm = lazy_map.Lazy_Map(
            sequence,
            lambda x: {"serial_number":serial_number, "run":run, "distance_ft":distance, "iq":x}
        )

        return lm





if __name__ == "__main__":
    import time
    print("Cheesed to meet you")

    oracle_sequence = ORACLE_Sequence(
        # desired_distances=[14,2,56],
        # desired_runs=[1],
        # desired_serial_numbers=["3123D52","3123D65","3123D79"],
        desired_serial_numbers=ALL_SERIAL_NUMBERS,
        desired_distances=ALL_DISTANCES_FEET,
        desired_runs=ALL_RUNS,
        window_length=256,
        window_stride=1,
        num_examples_per_device=100000,
        # num_examples_per_device=100,
        # seed=int(1337
        seed=int(time.time()),
        max_cache_size=100000*16
        # max_cache_size=1
    )

    
    import timeit
# <average time in seconds> = timeit.timeit(lambda: <muh shit>, number=<number of iterations>)
    def iterate():
        for i in oracle_sequence:
            pass
    
    print(timeit.timeit(lambda: iterate(), number=1))
    print(timeit.timeit(lambda: iterate(), number=100))
    # print(timeit.timeit(lambda: iterate(), number=1))

    print(len(oracle_sequence))