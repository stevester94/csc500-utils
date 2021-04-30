#! /usr/bin/python3
import io
import pprint
import re
import numpy as np
import sys


pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint


def _get_file_size(path):
    size = 0
    with open(path, "rb") as handle:
        handle.seek(0, io.SEEK_END)
        size = handle.tell()
    
    return size


def _2D_IQ_from_bytes(bytes):
    iq_2d_array = np.frombuffer(bytes, dtype=np.single)
    iq_2d_array = iq_2d_array.reshape((2,int(len(iq_2d_array)/2)), order="F")

    return iq_2d_array

def _metadata_from_path(path):
    match  = re.search("day-([0-9]+)_transmitter-([0-9]+)_transmission-([0-9]+)", path)
    (day, transmitter_id, transmission_id) = match.groups()

    return {
        "day": int(day),
        "transmitter_id": int(transmitter_id),
        "transmission_id": int(transmission_id)
    }

def build_element_from_path_and_offset(path, offset, symbol_size=384):
    with open(path, "rb") as handle:
        handle.seek(offset)

        b = handle.read(symbol_size)

        iq_2d_array = _2D_IQ_from_bytes(b)

        symbol_index_within_file = int(offset / symbol_size)

        metadata = _metadata_from_path(path)

        element =  {
            'transmitter_id': metadata["transmitter_id"],
            'day': metadata["day"],
            'transmission_id': metadata["transmission_id"],
            'frequency_domain_IQ': iq_2d_array,
            'frame_index':    -1,
            'symbol_index': symbol_index_within_file,
        }

        return element

class Binary_OFDM_Symbol_Random_Accessor():
    def __init__(self, 
        paths,
        symbol_size=384):

        print("BOSRA INIT")

        self.symbol_size = symbol_size

        self.containers = []
        running_offset = 0
        for p in paths:

            c = {
                "path": p,
                "size_bytes": _get_file_size(p),
                "metadata": _metadata_from_path(p),
                "start_offset": running_offset
            }

            self.containers.append(c)

            running_offset += c["size_bytes"]

        total_bytes = sum( [c["size_bytes"] for c in self.containers] )


        if total_bytes % self.symbol_size != 0:
            print("Total Bytes:", total_bytes)
            print("Symbol Size:", symbol_size)
            raise Exception("Total bytes is not divisible by symbol size")

        self.cardinality = int(total_bytes / self.symbol_size)

        self.offset_lookup_list = [c["start_offset"] for c in self.containers]

    # (<which file handle contains the index>, <the offset in that file for the index>)
    def _get_container_and_offset_of_index(self,index):
        offset = index * self.symbol_size

        # Find the indices into a sorted array a such that, if the corresponding elements in v were 
        #    inserted before the indices, the order of a would be preserved.
        idx = np.searchsorted(self.offset_lookup_list, offset) - 1
        c = self.containers[idx]

        if idx == len(self.offset_lookup_list):
            if offset > c["start_offset"] + c["size_bytes"]:
                print(self.offset_lookup_list)
                print("offset:", offset)
                print("Requested index:", index)
                print("Lookup index:", idx)
                print("cardinality", self.get_cardinality())
                raise IndexError("index out of range")
        
        return (c, offset-c["start_offset"])

    def get_path_and_offset_of_index(self, index):
        c = self._get_container_and_offset_of_index(index)
        return c[0]["path"], c[1]


    def __getitem__(self, index):
        container, offset = self._get_container_and_offset_of_index(index)

        with open(container["path"], "rb") as handle:
            handle.seek(offset)

            b = handle.read(self.symbol_size)

            iq_2d_array = _2D_IQ_from_bytes(b)

            symbol_index_within_file = offset / self.symbol_size
            
            return {
                'transmitter_id': container["metadata"]["transmitter_id"],
                'day': container["metadata"]["day"],
                'transmission_id': container["metadata"]["transmission_id"],
                'frequency_domain_IQ': iq_2d_array,
                'frame_index':    -1,
                'symbol_index': symbol_index_within_file,
            }

    def get_cardinality(self):
        return self.cardinality



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
    bosra = Binary_OFDM_Symbol_Random_Accessor(files)

    while True:
        count = 0
        for i in range(bosra.get_cardinality()):
            count += 1

        print(count)