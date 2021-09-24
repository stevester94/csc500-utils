#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from rotated_mnist_dataset import Rotated_MNIST_DS

class CIDA_MNIST_DS(torch.utils.data.Dataset):
    def __init__(self, seed, num_domains, min_rotation_degrees, max_rotation_degrees, num_examples_per_domain) -> None:
        super().__init__()

        assert(min_rotation_degrees >= 0)
        assert(max_rotation_degrees <= 360)

        self.rng = np.random.default_rng(seed)

        self.data = []

        """
        Generate ranges to get something like
        0,120
        120,240
        240,360
        """
        domain_ranges = []
        temp_domain_ranges = np.linspace(min_rotation_degrees, max_rotation_degrees, num_domains+1)
        for i, _ in enumerate(temp_domain_ranges):
            if i == len(temp_domain_ranges)-1:
                continue
            domain_ranges.append(
                (temp_domain_ranges[i], temp_domain_ranges[i+1])
            )
        
        for t, degree_range in enumerate(domain_ranges):
            ds = Rotated_MNIST_DS(seed, degree_range[0], degree_range[1])

            for rando in self.rng.choice(len(ds), size=num_examples_per_domain, replace=False):
                example = ds[rando]
                self.data.append(
                    (example[0], example[1], t)
                )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

        

if __name__ == "__main__":
    import random

    cida_mnist_ds = CIDA_MNIST_DS(1337, 2, 0, 180, 1000)

    l = list(cida_mnist_ds)
    random.shuffle(l)
    

    print(len(l))

    for x,y,t in l:

        figure, axis = plt.subplots(1, 1, figsize=(10,10))

        axis.imshow(x[0])
        axis.set_title(f'Label: {y}\nDomain: {t:.0f}')

        plt.show()