#! /usr/bin/env python3

# Borrows heavily from https://github.com/hehaodele/CIDA.git

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


import steves_utils.utils_v2 

import os

ROOT_DATASET_DIR = steves_utils.utils_v2.get_datasets_base_path()

def download_mnist():
    from torchvision.datasets import MNIST
    root = ROOT_DATASET_DIR

    processed_folder = os.path.join(root, 'MNIST', 'processed')
    if not os.path.isdir(processed_folder):
        dataset = MNIST(root=root, download=True)

class Rotated_MNIST_DS(Dataset):
    def __init__(self, root, seed, rotation_min_degrees, rotation_max_degrees):
        processed_folder = os.path.join(root, 'MNIST', 'processed')
        data_file = 'training.pt'
        download_mnist()
        
        self.X, self.Y = torch.load(os.path.join(processed_folder, data_file))
        self.rotation_min_degrees = rotation_min_degrees
        self.rotation_max_degrees = rotation_max_degrees

        self.rng = np.random.default_rng(seed)

    def __getitem__(self, index):
        from torchvision.transforms.functional import rotate
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label, angle_degrees)
        """
        img, label = self.X[index], int(self.Y[index])

        img = Image.fromarray(img.numpy(), mode='L')

        angle = self.rng.random() * (self.rotation_max_degrees - self.rotation_min_degrees) + self.rotation_min_degrees

        img = rotate(img, angle)
        img = transforms.ToTensor()(img).to(torch.float)

        x = img
        y = label


        return x, y, angle

    def __len__(self):
        return len(self.X)


if __name__ == "__main__":
    rmnist = Rotated_MNIST_DS(ROOT_DATASET_DIR, 2000, 0,45)

    img, label, angle = rmnist[0]



    figure, axis = plt.subplots(1, 1, figsize=(18,1.5))

    axis.imshow(img[0])

    print(label, angle)
    print(len(rmnist))

    plt.show()

    # for i in rmnist:
        # print("yeet")




    # for i in range(8):
    #     dataset = Rotated_MNIST_DS(ROOT_DATASET_DIR, rotate_angle=(i*45,i*45+45))
    #     if i == 0:
    #         dname = 'Source'
    #     else:
    #         dname = f'Sub Target #{i}'
    #     print(dname)
    #     fig, ax = plt.subplots(1, 10, figsize=(18,1.5))
    #     for j in range(10):
    #         img, label, angle, _ = dataset[j]
    #         angle = angle[0] * 360
    #         ax[j].imshow(img[0])
    #         ax[j].set_title(f'Label: {label}\nRot: {angle:.0f}')
    #     plt.show()
    #     plt.close()