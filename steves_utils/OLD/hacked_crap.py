#! /usr/bin/python3

from binary_symbol_dataset_accessor import BinarySymbolDatasetAccessor
from datasetaccessor import SymbolDatasetAccessor
from binary_random_accessor import Binary_OFDM_Symbol_Random_Accessor
from timeit import Timer
import sys
import tensorflow as tf



bsda = BinarySymbolDatasetAccessor(
    1337,
    day_to_get=[1],
    transmitter_id_to_get=[10,11],
    #transmitter_id_to_get=[11],
    #transmission_id_to_get=[1,2],
    transmission_id_to_get=[1],
)

bsda_ds = bsda.get_dataset()

print(bsda.get_dataset_cardinality())

def get_all_in_ds(ds):
    count = 0
    for e in ds:
        count += 1
    print("Number elements in dataset", count)


def get_all_in_generator(g):
    count = 0
    for e in g():
        count += 1
    print("Number elements in generator", count)

def custom_generator():
    paths = [   
        '/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/bin/day-1_transmitter-11_transmission-1.bin',
        '/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/bin/day-1_transmitter-10_transmission-1.bin'
    ]

    b = Binary_OFDM_Symbol_Random_Accessor(paths, 1337)

    for e in b.generator():
        yield (e["frequency_domain_IQ"], e["transmitter_id"])


def braindead_generator():
    for i in range(515368):
        yield tf.constant(
            0, dtype=tf.int64, shape=(), name='Const'
        )

ds = tf.data.Dataset.from_generator(
    # custom_generator,
    #  output_signature=(
    #      tf.TensorSpec(shape=(2,48), dtype=tf.float32),
    #      tf.TensorSpec(shape=(), dtype=tf.int64)
    #  )

    braindead_generator,
     output_signature=(
         tf.TensorSpec(shape=(), dtype=tf.int64)
     )
)


gen_timer = Timer(lambda: get_all_in_generator(custom_generator))
ds_timer = Timer(lambda: get_all_in_ds(ds))

print("Timing")
# print("bsda: ", bsda_timer.timeit(number=1))
# print("sda: ", sda_timer.timeit(number=1))
print("generator: ", gen_timer.timeit(number=1))
print("dataset: ", ds_timer.timeit(number=1))

# Times
# 
# No Parallelism in bsda
# bsda:  135.9338641679933
# sda:  41.72036427700368

# No parallelism at all 
# sda:  41.72036427700368

# split bosra in bsda, parallelism=6
# bsda: 67

# split bosra in bsda, parallelism=1
# bsda: 137

# Ultra-cool dataset parallelism, parallelism=2
# bsda: quit after 2m53s (memory leak)

# Ultra-cool dataset parallelism, parallelism=2
# bsda: 137.33539364099852