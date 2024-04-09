import os
import torch
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import InterpolationMode


class OxfordPetDataset(torch.utils.data.Dataset):

    def __init__(self, root, mode="train", augmentation=False):
        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")
        self.filenames = self._read_split()  # read train/valid/test splits

        # transformations
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)]
        )

        print(f"Loading {self.mode} dataset | Total {len(self.filenames)} images.")
        # read images
        self.sample = []
        for filename in tqdm(self.filenames):
            image_path = os.path.join(self.images_directory, filename + ".jpg")
            mask_path = os.path.join(self.masks_directory, filename + ".png")

            # convert data to tensor
            image = np.array(Image.open(image_path).convert("RGB"))
            image = transforms.functional.to_image(image)  # convert from HWC to CHW

            trimap = np.array(Image.open(mask_path))
            mask = self._preprocess_mask(trimap)

            trimap = torch.tensor(trimap, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

            # Apply augmentation
            if augmentation and mode != "test":
                image, mask, trimap = self.apply_augmentation(image, mask, trimap)

            # Apply transformation
            image = self.image_transform(image)
            mask = self.mask_transform(mask)
            trimap = self.mask_transform(trimap)

            self.sample.append(dict(image=image, mask=mask, trimap=trimap))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.sample[idx]

    def apply_augmentation(self, image_tensor, mask_tensor, trimap_tensor):
        # Apply random horizontal flip
        if torch.rand(1) < 0.5:
            image_tensor = F.hflip(image_tensor)
            mask_tensor = F.hflip(mask_tensor)
            trimap_tensor = F.hflip(trimap_tensor)

        # Apply random vertical flip
        if torch.rand(1) < 0.5:
            image_tensor = F.vflip(image_tensor)
            mask_tensor = F.vflip(mask_tensor)
            trimap_tensor = F.vflip(trimap_tensor)

        # Apply random rotation
        angle = torch.FloatTensor(1).uniform_(-30, 30)
        image_tensor = F.rotate(image_tensor, angle)
        mask_tensor = F.rotate(mask_tensor, angle)
        trimap_tensor = F.rotate(trimap_tensor, angle)

        return image_tensor, mask_tensor, trimap_tensor

    @staticmethod
    def _preprocess_mask(mask):
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


def load_dataset(data_path, mode):
    # implement the load dataset function here

    return NotImplemented
