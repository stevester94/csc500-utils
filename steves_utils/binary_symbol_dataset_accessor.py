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

from steves_utils.binary_random_accessor import Binary_OFDM_Symbol_Random_Accessor

pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint

def pool_worker_init(paths):
    global BOSRA
    BOSRA = Binary_OFDM_Symbol_Random_Accessor(paths)

def pool_worker_process(index):
    global BOSRA
    return BOSRA[index]

class BinarySymbolDatasetAccessor():
    def __init__(
        self,
        seed,
        batch_size,
        num_class_labels,
        train_val_test_splits=(0.6, 0.2, 0.2),
        bin_path="/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/bin",
        day_to_get="All",
        transmitter_id_to_get="All",
        transmission_id_to_get="All",
        num_workers=10
    ):
        self.day_to_get = day_to_get
        self.transmitter_id_to_get = transmitter_id_to_get
        self.transmission_id_to_get = transmission_id_to_get
        self.seed = seed
        self.batch_size = batch_size
        self.num_class_labels = num_class_labels
        self.train_val_test_splits = train_val_test_splits

        self.rng = np.random.default_rng(self.seed)


        self.paths = self.filter_datasets(self.get_binaries_in_dir(bin_path))

        if len(self.paths) == 0:
            print("No paths remained after filtering. Time to freak out!")
            sys.exit(1)

        self.rng.shuffle(self.paths)
        
        self.bosra = Binary_OFDM_Symbol_Random_Accessor(self.paths, max_file_descriptors=0)
        self.cardinality = self.bosra.get_cardinality()

        # We generate our own indices based on the seed
        print("Generating our random indices")
        indices = np.arange(0, self.cardinality)
        self.rng.shuffle(indices)

        # Build the train/val/test indices lists
        train_size = int(self.cardinality * train_val_test_splits[0])
        val_size  = int(self.cardinality * train_val_test_splits[1])
        test_size  = int(self.cardinality * train_val_test_splits[2])

        self.indices       = indices
        self.train_indices = indices[:train_size]
        self.val_indices  = indices[train_size:train_size+val_size]
        self.test_indices  = indices[train_size+val_size:]

        # Initialize the WORKERS!
        print("Initializing workers")
        self.worker_pool = mp.Pool(num_workers, pool_worker_init, (self.paths,))


        print("Ready")

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


    def get_paths(self):
        return self.paths
    
############################################

    def get_total_dataset_cardinality(self):
        return self.cardinality
        
    def get_train_dataset_cardinality(self):
        return len(self.train_indices)

    def get_val_dataset_cardinality(self):
        return len(self.val_indices)

    def get_test_dataset_cardinality(self):
        return len(self.test_indices)


    def _index_generator(self, indices, repeat, shuffle):
        while True:
            if shuffle:
                self.rng.shuffle(indices)

            yield from indices

            if not repeat:
                return

    def train_generator(self, repeat=True, shuffle=True):
        pool_imap = self.worker_pool.imap(pool_worker_process, self._index_generator(self.train_indices, repeat, shuffle))

        # yield from pool_imap

        yield from self.batch_generator_from_generator(pool_imap, self.batch_size)


    def val_generator(self, repeat=True, shuffle=False):
        pool_imap = self.worker_pool.imap(pool_worker_process, self._index_generator(self.val_indices, repeat, shuffle))
        yield from self.batch_generator_from_generator(pool_imap, self.batch_size)

    def test_generator(self, repeat=False, shuffle=False):
        pool_imap = self.worker_pool.imap(pool_worker_process, self._index_generator(self.test_indices, repeat, shuffle))
        yield from self.batch_generator_from_generator(pool_imap, self.batch_size)

    # Does not drop remainder, makes a baby set
    def batch_generator_from_generator(self, gen, batch_size):
        exhausted = False
        while True:
            x = []
            y = []
            for i in range(batch_size):
                try:
                    e = next(gen)
                except StopIteration:
                    exhausted = True

                x.append( e["frequency_domain_IQ"] )
                y.append( e["transmitter_id"] )


            try:
                x = tf.convert_to_tensor(x, dtype=tf.float32)
                y = tf.convert_to_tensor(y, dtype=tf.int64)
            except:
                pprint(x)
                pprint(y)
                raise

        
            yield (
                x,
                tf.one_hot(tf.convert_to_tensor(y, dtype=tf.int64), self.num_class_labels) # One hot is quite slow, should be called on the top level tens
            )

            # yield (x,y)
            # if exhausted: return
            if exhausted: raise StopIteration
    
    def dataset_from_generator(self, generator):
        ds = tf.data.Dataset.from_generator(
            generator,
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

if __name__ == "__main__":
    RANGE   = 12
    BATCH   = 100
    EPOCHS  = 5
    DROPOUT = 0.5 # [0,1], the chance to drop an input

    bsda = BinarySymbolDatasetAccessor(
        seed=int(sys.argv[1]),
        batch_size=BATCH,
        num_class_labels=RANGE,
        bin_path="../../csc500-dataset-preprocessor/bin/",
        # day_to_get=[1,2,3,4,5,6,7,8],
        # transmitter_id_to_get=[10,11],
        # transmission_id_to_get=[2],

        # day_to_get=[1],
        # transmitter_id_to_get=[10],

        # This reveals that invalid argument issue
        transmission_id_to_get=[2],
        day_to_get=[1],
        transmitter_id_to_get=[10,11],
    )

    bosra = bsda.bosra

    
    ds = bsda.dataset_from_generator(bsda.train_generator)

    count = 0
    last_time = time.time()
    total_count=0
    # for i in bsda.train_generator():
    #     count += 1
    for i in ds.batch(BATCH):
        count += BATCH
        
        if count % (10000) == 0:
            items_per_sec = count / (time.time() - last_time)
            print("Items per second:", items_per_sec)
            last_time = time.time()
            count = 0



    sys.exit(1)

    print("GO")
    print(bsda.get_val_dataset_cardinality())
    # 24 seconds even with out tf convert
    count = 0
    last_time = time.time()
    total_count=0
    for i in bsda.test_generator():
        total_count += BATCH
        count += 1

        if count % (10000 / BATCH) == 0:
            items_per_sec = count / (time.time() - last_time)
            print("current:",total_count,"total:",bsda.get_test_dataset_cardinality())
            # print("Batches per second:", items_per_sec)
            # print("Items per second:", items_per_sec*BATCH)
            last_time = time.time()
            count = 0
        # if total_count  >= bsda.get_val_dataset_cardinality():
        #     break

    # print(bsda.get_train_dataset_cardinality())
    # print(total_count)

    # 5.8 seconds
    # count = 0
    # last_time = time.time()
    # total_count = 0
    # for i in bsda.train_indices:
    #     count += 1
    #     total_count += 1

    #     fuck = bosra[i]


    #     if count % (10000) == 0:
    #         items_per_sec = count / (time.time() - last_time)

    #         print("Items per second:", items_per_sec)
    #         last_time = time.time()
    #         count = 0
    
    # print(bsda.get_train_dataset_cardinality())
    # print(total_count)



    # import threading

    # def GO(indices):
    #     # BOSRA ONLY SPEED TEST!
    #     bosra = bsda.bosra
    #     print("GO")
    #     while True:
    #         s = 0.0
    #         for i in indices:
    #             e = bosra[i]
    #             s += e["transmission_id"]

    #         print(s)
    
    # indices = []
    # parallelism = 3 # Need an extra one just because we aren't appending non-complete index slices
    # last_start = 0
    # chunk_length = int(len(bsda.indices) / parallelism)
    # while last_start + chunk_length < len(bsda.indices):
    #     indices.append(bsda.indices[last_start:last_start+chunk_length])
    #     last_start += chunk_length
    
    # print(len(indices))

    # threads = []
    # for i in indices:
    #     t = threading.Thread(target=GO, args=(i,))
    #     t.start()
    #     threads.append(t)
    #     print("loop")

    # bsda.indices

########################################################################################33
# Like 20k items/sec!
    # def read_file_at_offset(args):
    #     with open(args[0], "rb") as f:
    #         f.seek(args[1])

    #         buf = f.read(384)

    #         return buf

    # random.seed(sys.argv[1])

    # def offset_generator():
    #     smallest_file_in_bytes = 100017792

    #     # while True:
    #     for i in range(int(len(bsda.indices) / len(bsda.paths))):
    #         for p in bsda.paths:
    #             yield (p, random.randint(0, smallest_file_in_bytes))
    

    # print("GO")

    # with mp.Pool(10) as pool:
    #     i = pool.imap(read_file_at_offset, offset_generator())

    #     count = 0
    #     last_time = time.time()
    #     for buf in i:
    #         count += 1

    #         if count % 10000 == 0:
    #             items_per_sec = count / (time.time() - last_time)

    #             print("Items per second:", items_per_sec)
    #             last_time = time.time()
    #             count = 0

########################################################################################33
# ~2k items/sec
    # with mp.Pool(8) as pool:
    #     i = pool.imap(bosra.__getitem__, bsda.indices)

    #     count = 0
    #     last_time = time.time()
    #     for buf in i:
    #         count += 1

    #         if count % 10000 == 0:
    #             items_per_sec = count / (time.time() - last_time)

    #             print("Items per second:", items_per_sec)
    #             last_time = time.time()
    #             count = 0

########################################################################################

    # # Equivalent performace with the gen, about 22k/sec
    # def gen(indices, bosra):
    #     for i in indices:
    #         yield bosra.get_path_and_offset_of_index(i)

    # # def proc(args):
    # #     path = args[0]
    # #     offset = args[1]
    # #     with open(path, "rb") as handle:
    # #         handle.seek(offset)

    # #         b = handle.read(388)

    # #         return b

    # def proc(args):
    #     return build_element_from_path_and_offset(args[0], args[1], 384)

    # with mp.Pool(20) as pool:
    #     # i = pool.imap(bosra._get_container_and_offset_of_index, bsda.indices)
    #     # i = pool.imap(bosra.dumb, bsda.indices)
    #     # i = pool.imap(proc, gen(bsda.indices, bosra))
    #     # i = gen(bsda.indices, bosra)
    #     # i = pool.imap(proc, targets)
    #     i = pool.imap(proc, gen(bsda.indices, bosra))

    #     count = 0
    #     last_time = time.time()
    #     for buf in i:
    #         count += 1

    #         if count % 10000 == 0:
    #             items_per_sec = count / (time.time() - last_time)

    #             print("Items per second:", items_per_sec)
    #             last_time = time.time()
    #             count = 0

########################################################################################
# ~22k items/sec
    # count = 0
    # last_time = time.time()
    # for i in bsda.indices:
    #     X = bosra.get_path_and_offset_of_index(i)
    #     count += 1

    #     if count % 10000 == 0:
    #         items_per_sec = count / (time.time() - last_time)

    #         print("Items per second:", items_per_sec)
    #         last_time = time.time()
    #         count = 0














    ###########################################
    #GRAVEYARD
# def split(a, n):
#     k, m = divmod(len(a), n)
#     l = (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
#     l = list(l)
#     return [thing for thing in l if len(thing) > 1]


# def _one_hot_encoder_generator(bosra_generator, num_class_labels):
#     for e in bosra_generator:
        
#         yield (
#             e[0],
#             tf.one_hot(tf.convert_to_tensor(e[1], dtype=tf.int64), num_class_labels) # One hot is quite slow, should be called on the top level tens
#         )