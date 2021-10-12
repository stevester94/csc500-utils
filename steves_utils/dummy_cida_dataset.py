#! /usr/bin/env python3

import numpy as np
import random
import torch

class Dummy_CIDA_Dataset(torch.utils.data.Dataset):
    def __init__(self, x_shape, domains:list, num_classes:int, num_unique_examples_per_class:int, normalize_domain:int=-1) -> None:
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

        if normalize_domain != -1:
            domains = map(lambda k: k/normalize_domain, domains)

        for u in domains:
            for y in range(num_classes):
                for i in range(num_unique_examples_per_class):
                    x = np.array(x_source * y * u, dtype=np.single)
                    u_array = np.array([u], dtype=np.single)

                    examples.append((x,y,u_array))

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