#! /usr/bin/env python3

import numpy as np
from math import floor

from numpy.testing._private.utils import assert_equal

class File_As_Windowed_Sequence:
    """
    Creates an indexable sequence out of a file given the window size and the stride size.

    """
    def __init__(self, path:str, window_length:int, stride:int, numpy_dtype:np.dtype) -> None:
        self.memmap = np.memmap(path, numpy_dtype)
        # self.view = np.lib.stride_tricks.sliding_window_view(self.memmap, window_length, stride)

        self.window_length = window_length
        self.stride = stride

        self.len = floor((len(self.memmap) - self.window_length) / self.stride) + 1

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if index >= self.len or index < 0:
            raise IndexError
        return self.memmap[index*self.stride : index*self.stride+self.window_length]
        
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

    


if __name__ == "__main__":
    import unittest
    import tempfile

    TEST_DTYPE = np.single
    TEST_BUFFER_NUM_ITEMS = 10000
    class test_File_As_Windowed_Sequence(unittest.TestCase):
        @classmethod
        def setUpClass(self) -> None:
            self.dtype = TEST_DTYPE
            self.num_items = TEST_BUFFER_NUM_ITEMS



            self.f = tempfile.NamedTemporaryFile("w+b")
            self.source = np.random.default_rng().integers(0, 100, self.num_items).astype(self.dtype)

            self.f.write(self.source.tobytes())
            self.f.seek(0)


        # @classmethod
        # def tearDownClass(self) -> None:
        #     pass

        def test_len(self):
            buf = np.frombuffer(self.f.read(), dtype=self.dtype)

            self.assertEqual(len(buf), self.num_items)
            self.assertEqual(len(buf), len(self.source))
        
        def test_big_window(self):
            faws = File_As_Windowed_Sequence(self.f.name, window_length=self.num_items, stride=30, numpy_dtype=self.dtype)

            self.assertTrue(np.array_equal(faws[0], self.source))
            
        def test_window_beginnings_are_true_to_source(self):
            stride = 5
            faws = File_As_Windowed_Sequence(self.f.name, window_length=100, stride=stride, numpy_dtype=self.dtype)

            built_up = []
            for i in faws:
                built_up.extend(i[:stride])

            built_up = np.array(built_up, dtype=self.dtype)
            
        

    unittest.main()

    path = "/mnt/wd500GB/CSC500/csc500-super-repo/datasets/automated_windower/windowed_EachDevice-200k_batch-100_stride-1_distances-2/test/batch-100_part-0.tfrecord_ds"


    faws = File_As_Windowed_Sequence(path, window_length=256, stride=30, numpy_dtype=np.float)

    print(len(faws))

    
