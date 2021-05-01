#! /usr/bin/python3

from datasetaccessor import SymbolDatasetAccessor
from timeit import Timer
import sys
import time
import gc




def fun():
    import tensorflow as tf
    from fixed_length_accessor import FixedLengthDatasetAccessor

    flda = FixedLengthDatasetAccessor(bin_path="/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/conversion/test")
    ds = flda.get_dataset()
    ds = ds.take(10000)
    ds = ds.shuffle(10000)
    ds = ds.cache("fug")

    for e in ds:
        pass

    print("Done")


fun()

while True:
    gc.collect()
    time.sleep(1)
# print(ds1.cardinality())

# print("Shuffling")
# ds1 = ds1.shuffle(ds1.cardinality())

# print("Counting")
# count = 0
# for e in ds1:
#     count += 1


