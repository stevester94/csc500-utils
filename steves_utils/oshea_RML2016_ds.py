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
        snrs_to_get:list=None,
        normalize_snr:bool=False) -> None:
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

        self.snrs = snrs

        if normalize_snr:
            min_snr = min(self.get_snrs())
            max_snr_after_min = max(self.get_snrs()) - min_snr
            normalizer_func = lambda snr: (snr-min_snr)/max_snr_after_min
        

        data = []  
        lbl = []
        for mod in mods:
            for snr in snrs:

                if snrs_to_get == None or snr in snrs_to_get:
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

    # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
    def get_snrs(self):
        return self.snrs
        


if __name__ == "__main__":
    ds = OShea_RML2016_DS(normalize_snr=True)

    s = list(set([float(x[2]) for x in ds]))
    s.sort()

    print(s)

    
