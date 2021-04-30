#! /usr/bin/python3

from binary_symbol_dataset_accessor import BinarySymbolDatasetAccessor
from datasetaccessor import SymbolDatasetAccessor
from binary_random_accessor import Binary_OFDM_Symbol_Random_Accessor
from timeit import Timer
import sys
import tensorflow as tf

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

def gen():
    while True:
        yield 1


def dataset_from_generator( generator):
    ds = tf.data.Dataset.from_generator(
        generator,
        output_types=(
            tf.int64
        ),
        output_shapes=(
            (None)
        )
    )

    return ds


ds = dataset_from_generator(gen)
ds = ds.prefetch(1000)
ds = ds.batch(500)
speed_test(ds, batch_size=500)