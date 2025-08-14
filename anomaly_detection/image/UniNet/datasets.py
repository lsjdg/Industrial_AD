import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import glob

# MVTecAD classes
CLASSES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecDataset(Dataset):
    def __init__(
        self,
        c,  # argparser
        class_name="bottle",
        is_train=True,
        resize=256,
    ):
        # check availability of class name
        assert class_name in CLASSES, f"class {class_name} is not available."

        self.path = "../datasets/MVTecAD"
        self.class_name = class_name
        self.resize = resize
        self.input_size = (c.image_size, c.image_size)
        self.phase = "train" if is_train else "test"
        self.img_dir = os.path.join(self.path, self.class_name, self.phase)
        self.gt_dir = os.path.join(self.path, self.class_name, "ground_truth")

        # load dataset
        self.x, self.y, self.mask = self.load_dataset()

        # set transforms
        if is_train:
            self.transform_x = T.Compose(
                [
                    T.Resize(self.input_size, T.InterpolationMode.LANCZOS),
                    T.ToTensor(),
                ]
            )
        else:
            self.transform_x = T.Compose(
                [
                    T.Resize(self.input_size, T.InterpolationMode.LANCZOS),
                    T.ToTensor(),
                ]
            )
            self.transform_mask = T.Compose(
                [
                    T.Resize(self.input_size, T.InterpolationMode.NEAREST),
                    T.ToTensor(),
                ]
            )
            self.normalize = T.Compose(
                [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x)

        # handle grayscale imgs
        if self.class_name in ["zipper", "screw", "grid"]:
            x = np.expand_dims(np.array(x), axis=2)  # (H, W, C)
            x = np.concatenate([x, x, x], axis=2)

            x = Image.fromarray(x.astype("uint8")).convert("RGB")

        x = self.normalize(self.transform_x(x))

        if y == 0:
            mask = torch.zeros(self.input_size)
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset(self):
        img_paths = list()
        gt_paths = list()
        labels = list()
        types = list()

        defect_types = os.listdir(self.img_dir)

        for defect_type in defect_types:
            if defect_type == "good":
                img_paths.extend(
                    glob.glob(os.path.join(self.img_dir, defect_type) + "/*")
                )
                gt_paths.extend([None] * len(img_paths))
                labels.extend([0] * len(img_paths))
            else:
                img_paths.extend(
                    glob.glob(os.path.join(self.img_dir, defect_type) + "/*")
                )
                img_paths.sort()

                gt_paths.extend(
                    glob.glob(os.path.join(self.gt_dir, defect_type) + "/*")
                )
                gt_paths.sort()
                labels.extend([1] * len(img_paths))

        assert len(img_paths) == len(
            labels
        ), "Something wrong with test and ground truth pair!"

        return img_paths, labels, gt_paths, types
