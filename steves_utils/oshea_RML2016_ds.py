#! /usr/bin/env python3

import pickle
import numpy as np
import torch


class OShea_RML2016_DS(torch.utils.data.Dataset):
    """
    snrs: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]

    modulation_mapping = {
        'AM-DSB': 0,
        'QPSK'  : 1,
        'BPSK'  : 2,
        'QAM64' : 3,
        'CPFSK' : 4,
        '8PSK'  : 5,
        'WBFM'  : 6,
        'GFSK'  : 7,
        'AM-SSB': 8,
        'QAM16' : 9,
        'PAM4'  : 10,
    }
    """
    def __init__(self, path:str="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/RML2016.10a_dict.pkl", 
        snrs_to_get:list=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2],
        normalize_snr:tuple=None) -> None:
        """
        args:
            domain_configs: {
                "domain_index":int,
                "min_rotation_degrees":float,
                "max_rotation_degrees":float,
                "num_examples_in_domain":int,
            }
        """
        super().__init__()

        self.modulation_mapping = {
            'AM-DSB': 0,
            'QPSK'  : 1,
            'BPSK'  : 2,
            'QAM64' : 3,
            'CPFSK' : 4,
            '8PSK'  : 5,
            'WBFM'  : 6,
            'GFSK'  : 7,
            'AM-SSB': 8,
            'QAM16' : 9,
            'PAM4'  : 10,
        }

        Xd = pickle.load(open(path,'rb'), encoding="latin1")
        self.Xd = Xd
        snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

        if normalize_snr is not None:
            assert(normalize_snr[0] < normalize_snr[1])
            assert(normalize_snr[0] <= min(snrs_to_get))
            assert(normalize_snr[1] >= max(snrs_to_get))

            min_snr = normalize_snr[0]
            max_snr_after_min = normalize_snr[1] - min_snr
            normalizer_func = lambda snr: (snr-min_snr)/max_snr_after_min
        

        data = []  
        lbl = []
        for mod in mods:
            for snr in snrs:

                if snr in snrs_to_get:
                    for x in Xd[(mod,snr)]:
                        if normalize_snr:
                            snr = normalizer_func(snr)

                        data.append(
                            (
                                x.astype(np.single),
                                self.modulation_to_int(mod),
                                np.array([snr], dtype=np.single)
                            )
                        )

        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def modulation_to_int(self, modulation:str):
        return self.modulation_mapping[modulation]

    # 
    @classmethod
    def get_snrs(cls):
        return [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
        


if __name__ == "__main__":
    import unittest
    import random
    import itertools

    LEN_SEQUENCE = 100000
    MAX_CACHE_SIZE = 1000

    class test_OShea_RML2016_DS(unittest.TestCase):
        @classmethod
        def setUpClass(self) -> None:
            self.ds = OShea_RML2016_DS(normalize_snr=(-20,18))

        def test_normalization(self):
            source_snrs = [-18, -12, -6, 0, 6, 12, 18]
            target_snrs = [2, 4, 8, 10, -20, 14, 16, -16, -14, -10, -8, -4, -2]



            normalized_source_snrs = list(set([float(x[2]) for x in OShea_RML2016_DS(normalize_snr=(-20, 18), snrs_to_get=source_snrs)]))
            normalized_source_snrs.sort()

            print(normalized_source_snrs)

            normalized_target_snrs = list(set([float(x[2]) for x in OShea_RML2016_DS(normalize_snr=(-20, 18), snrs_to_get=target_snrs)]))
            normalized_target_snrs.sort()

            print(normalized_target_snrs)

            # non_normalized_snrs = OShea_RML2016_DS.get_snrs()
            # normalized_snrs = OShea_RML2016_DS.get_snrs()
            # normalized_snrs = list(map(lambda snr: (snr - min()) / (max(OShea_RML2016_DS.get_snrs())-min(OShea_RML2016_DS.get_snrs()))))
            # self.snrs.sort()

            # for i, snr in enumerate(self.snrs):
            #     test = 
            #     self.assertAlmostEqual(s[i], test)
    

    unittest.main()
    
