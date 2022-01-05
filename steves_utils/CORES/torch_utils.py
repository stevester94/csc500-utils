#! /usr/bin/env python3


import numpy as np
import pickle
import os
from definitions import *
from utils import get_dataset_by_day_and_node_name, get_cores_dataset_path
from typing import List
import torch

class CORES_Torch_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        days_to_get:List[int],
        num_examples_per_node_per_day,
        nodes_to_get,
        seed:int,
        root_dir:str=get_cores_dataset_path(),
	    transform_func=None,
    ) -> None:
        self.ds = get_dataset_by_day_and_node_name(
            num_examples_per_node_per_day=num_examples_per_node_per_day,
            nodes_to_get=nodes_to_get,
            seed=seed,
            days_to_get=days_to_get,
            root_dir=root_dir
        )

        self.transform_func = transform_func

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        if self.transform_func != None:
            return self.transform_func(self.ds[idx])
        else:
            return self.ds[idx]

if __name__ == "__main__":
    from utils import ALL_NODES, ALL_DAYS

    ctd = CORES_Torch_Dataset(
        days_to_get=ALL_DAYS,
        num_examples_per_node_per_day=100,
        nodes_to_get=ALL_NODES,
        seed=1337,
	    # transform_func=None,
    )
    print(ctd[0])

    ctd = CORES_Torch_Dataset(
        days_to_get=ALL_DAYS,
        num_examples_per_node_per_day=100,
        nodes_to_get=ALL_NODES,
        seed=1337,
	    transform_func=lambda ex: (ex["day"], ex["node_name"])
    )
    print(ctd[0])

