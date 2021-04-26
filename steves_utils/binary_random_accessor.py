#! /usr/bin/python3
import io
import pprint
import re
import numpy as np
import tensorflow as tf


pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint


def _get_file_size(path):
    size = 0
    with open(path, "rb") as handle:
        handle.seek(0, io.SEEK_END)
        size = handle.tell()
    
    return size

class Binary_OFDM_Symbol_Random_Accessor():
    def __init__(self, 
        paths,
        seed, 
        train_eval_test_splits=(0.6, 0.2, 0.2),
        symbol_size=384):

        print("BOSRA INIT")

        self.symbol_size = symbol_size
        self.seed = seed

        self.containers = []
        for p in paths:
            handle = open(p, "rb")

            self.containers.append({
                "handle": handle,
                "size_bytes": _get_file_size(p),
                "metadata": self._metadata_from_path(p)
            })

        total_bytes = sum( [c["size_bytes"] for c in self.containers] )
        assert(total_bytes % self.symbol_size == 0)
        self.cardinality = int(total_bytes / self.symbol_size)


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


    # (<which file handle contains the index>, <the offset in that file for the index>)
    def _get_container_and_offset_of_index(self,index):
        offset = index * self.symbol_size
        for f in self.containers:
            if offset < f["size_bytes"]:
                return (f, offset)
            else:
                offset -= f["size_bytes"]
        
        raise IndexError("index out of range")

    
    def _metadata_from_path(self, path):
        match  = re.search("day-([0-9]+)_transmitter-([0-9]+)_transmission-([0-9]+)", path)
        (day, transmitter_id, transmission_id) = match.groups()

        return {
            "day": int(day),
            "transmitter_id": int(transmitter_id),
            "transmission_id": int(transmission_id)
        }

    def _2D_IQ_from_bytes(self, bytes):
        iq_2d_array = np.frombuffer(bytes, dtype=np.single)
        iq_2d_array = iq_2d_array.reshape((2,int(len(iq_2d_array)/2)), order="F")

        return iq_2d_array

    def __getitem__(self, index):
        container, offset = self._get_container_and_offset_of_index(index)
        handle = container["handle"]

        handle.seek(offset)

        b = handle.read(self.symbol_size)
        iq_2d_array = self._2D_IQ_from_bytes(b)

        symbol_index_within_file = offset / self.symbol_size
        
        return {
            'transmitter_id': container["metadata"]["transmitter_id"],
            'day': container["metadata"]["day"],
            'transmission_id': container["metadata"]["transmission_id"],
            'frequency_domain_IQ': iq_2d_array,
            'frame_index':    -1,
            'symbol_index': symbol_index_within_file,
        }

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
            yield self[index]

    def eval_generator(self):
        for index in self.eval_indices:
            yield self[index]

    def test_generator(self):
        for index in self.test_indices:
            yield self[index]

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

    def dataset_from_BOSRA_generator(self, generator):
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
    files = [
        "../../csc500-dataset-preprocessor/bin/day-2_transmitter-4_transmission-1.bin", # 296559 elements
        "../../csc500-dataset-preprocessor/bin/day-2_transmitter-4_transmission-2.bin", # 262364 elements
    ]

    bosra = Binary_OFDM_Symbol_Random_Accessor(files, 1337)

    pprint(bosra[296559])
    pprint(bosra[0])
    pprint(bosra[10])
    pprint(bosra[2969])
    try:
        pprint(bosra[29600559])
    except:
        print("Failed succesfully lol")

    s = 0
    for i in bosra.train_generator():
        s += 1

    s = 0
    for i in bosra.eval_generator():
        s += 1

    s = 0
    for i in bosra.test_generator():
        s += 1
