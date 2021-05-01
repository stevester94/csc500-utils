#! /usr/bin/python3
import io
import pprint
import re
import numpy as np
import sys


pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint






class Binary_OFDM_Symbol_Random_Accessor():
    def __init__(self, 
        paths,
        symbol_size=384,
        max_file_descriptors=500):

        self.symbol_size = symbol_size

        self.containers = []
        running_offset = 0
        for idx,p in enumerate(paths):

            c = {
                "path": p,
                "size_bytes": self._get_file_size(p),
                "metadata": self._metadata_from_path(p),
                "start_offset": running_offset,
                "handle": None
            }

            # We do this under the assumption that the paths are randomized anyways.
            # It's simple, and it's better than nothing
            if idx < max_file_descriptors:
                c["handle"] = open(c["path"], "rb")

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
        idx = np.searchsorted(self.offset_lookup_list, offset, side="right") - 1
        c = self.containers[idx]

        # Edge case for the last file. Gotta make sure the offset actually falls within it
        if idx == len(self.offset_lookup_list) - 1:
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

        return self._element_from_container_and_offset(container, offset)

    def get_cardinality(self):
        return self.cardinality

    def _element_from_container_and_offset(self, container, offset):
        # Some containers may have their file handles pre-opened as an optimization
        # If not we open it here, then close once we are done
        if container["handle"] != None:
            handle = container["handle"]
        else:
            handle = open(container["path"], "rb")

        handle.seek(offset)

        b = handle.read(self.symbol_size)

        iq_2d_array = self._2D_IQ_from_bytes(b)

        symbol_index_within_file = int(offset / self.symbol_size)

        # metadata = _metadata_from_path(path)

        element =  {
            'transmitter_id': container["metadata"]["transmitter_id"],
            'day': container["metadata"]["day"],
            'transmission_id': container["metadata"]["transmission_id"],
            'frequency_domain_IQ': iq_2d_array,
            'frame_index':    -1,
            'symbol_index': symbol_index_within_file,
        }

        # Clean up FD
        if container["handle"] == None:
            handle.close()

        return element


    def _get_file_size(self, path):
        size = 0
        with open(path, "rb") as handle:
            handle.seek(0, io.SEEK_END)
            size = handle.tell()
        
        return size


    def _2D_IQ_from_bytes(self, bytes):
        iq_2d_array = np.frombuffer(bytes, dtype=np.single)
        iq_2d_array = iq_2d_array.reshape((2,int(len(iq_2d_array)/2)), order="F")

        return iq_2d_array

    def _metadata_from_path(self, path):
        match  = re.search("day-([0-9]+)_transmitter-([0-9]+)_transmission-([0-9]+)", path)
        (day, transmitter_id, transmission_id) = match.groups()

        return {
            "day": int(day),
            "transmitter_id": int(transmitter_id),
            "transmission_id": int(transmission_id)
        }


if __name__ == "__main__":
    bosra = Binary_OFDM_Symbol_Random_Accessor(files)

    while True:
        count = 0
        for i in range(bosra.get_cardinality()):
            count += 1

        print(count)