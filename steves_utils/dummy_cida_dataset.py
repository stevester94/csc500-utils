#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random
import torch

from steves_utils.rotated_mnist_dataset import Rotated_MNIST_DS

class Dummy_CIDA_Dataset(torch.utils.data.Dataset):
    def __init__(self, x_shape, domains:list, num_classes:int, num_unique_examples_per_class:int) -> None:
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

        examples = []

        x_source = np.ones(x_shape, dtype=np.float)

        for u in domains:
            for y in range(num_classes):
                for i in range(num_unique_examples_per_class):
                    x = np.array(x_source * y * u, dtype=np.single)
                    
                    examples.append((x,y,u))

        random.shuffle(examples)
        self.data = examples

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

        

if __name__ == "__main__":
    ds = Dummy_CIDA_Dataset(
        x_shape=[2,128],
        domains=[1,2,3,4,5,6],
        num_classes=10,
        num_unique_examples_per_class=5000
    )

    for d in ds:
        print(d)