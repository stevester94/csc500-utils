#! /usr/bin/python3

from binary_symbol_dataset_accessor import BinarySymbolDatasetAccessor
from datasetaccessor import SymbolDatasetAccessor
from binary_random_accessor import Binary_OFDM_Symbol_Random_Accessor
from timeit import Timer
import sys




paths = [   '/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/bin/day-1_transmitter-11_transmission-1.bin',
    '/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/bin/day-1_transmitter-10_transmission-1.bin']

bosra = Binary_OFDM_Symbol_Random_Accessor(paths, 1337)


bsda = BinarySymbolDatasetAccessor(
    1337,
    day_to_get=[1],
    transmitter_id_to_get=[10,11],
    #transmitter_id_to_get=[11],
    #transmission_id_to_get=[1,2],
    transmission_id_to_get=[1],
)
sda  = SymbolDatasetAccessor(tfrecords_path="/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/symbol_tfrecords" ,     
    day_to_get=[1],
    transmitter_id_to_get=[10,11],
    #transmitter_id_to_get=[11],
    #transmission_id_to_get=[1,2],
    transmission_id_to_get=[1],
)

bsda_ds = bsda.get_dataset()
sda_ds  = sda.get_dataset()

print(bsda.get_dataset_cardinality())

def get_all_in_ds(ds):
    count = 0
    for e in ds:
        count += 1
    print(count)

def get_all_in_bosra(b):
    count = 0
    for e in b.generator():
        count += 1
    print(count)





bsda_timer = Timer(lambda: get_all_in_ds(bsda_ds))
sda_timer = Timer(lambda: get_all_in_ds(sda_ds))
bosra_timer = Timer(lambda: get_all_in_bosra(bosra))

print("Timing")
# print("bsda: ", bsda_timer.timeit(number=1))
# print("sda: ", sda_timer.timeit(number=1))
print("bosra: ", bosra_timer.timeit(number=1))

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