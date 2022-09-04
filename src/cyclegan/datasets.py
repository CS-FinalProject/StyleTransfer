import glob
import os
import random

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode="train"):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(
            glob.glob(os.path.join(root, f"{mode}/A") + "/*.*"))
        self.files_B = sorted(
            glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

        self.random_crop = T.RandomCrop((128, 128))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(
            self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(
                self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(
                self.files_B[index % len(self.files_B)]))

        item_A = self.random_crop(item_A)
        item_B = self.random_crop(item_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
