#! /usr/bin/python3
import io
import pprint
import re
import numpy as np


pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint


def _get_file_size(path):
    size = 0
    with open(path, "rb") as handle:
        handle.seek(0, io.SEEK_END)
        size = handle.tell()
    
    return size

class Binary_OFDM_Symbol_Random_Accessor():
    def __init__(self, paths, seed, symbol_size=384):
        self.symbol_size = symbol_size

        self.containers = []
        for p in paths:
            handle = open(p, "rb")

            self.containers.append({
                "handle": handle,
                "size_bytes": _get_file_size(p),
                "metadata": self._metadata_from_path(p)
            })

        print("BOSRA INIT")

        total_bytes = sum( [c["size_bytes"] for c in self.containers] )
        assert(total_bytes % self.symbol_size == 0)
        self.cardinality = int(total_bytes / self.symbol_size)


        # We generate our own indices based on the seed
        rng = np.random.default_rng(seed)
        self.indices = np.arange(0, self.cardinality)
        rng.shuffle(self.indices)



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
            "day": day,
            "transmitter_id": transmitter_id,
            "transmission_id": transmission_id
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

    def get_dataset_cardinality(self):
        return self.cardinality

    def generator(self):
        for index in self.indices:
            yield self[index]


if __name__ == "__main__":
    files = [
        "../../csc500-dataset-preprocessor/bin/day-2_transmitter-4_transmission-1.bin", # 296559 elements
        "../../csc500-dataset-preprocessor/bin/day-2_transmitter-4_transmission-2.bin", # 262364 elements
    ]

    bosra = Binary_OFDM_Symbol_Random_Accessor(files, [10], 384)

    pprint(bosra[296559])
    pprint(bosra[0])
    pprint(bosra[10])
    pprint(bosra[2969])
    try:
        pprint(bosra[29600559])
    except:
        print("Failed succesfully lol")

    s = 0
    for i in bosra.generator():
        s += 1
    
    print(s)

    for i in bosra.generator():
        s += 1
    
    print(s)

