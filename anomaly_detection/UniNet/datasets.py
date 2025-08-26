from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="importlib._bootstrap"
)


def loading_dataset(c, dataset_name):
    train_dataloader, test_dataloader = None, None

    if dataset_name == "MVTecAD" and c.setting == "oc":
        train_data = MVTecDataset(c, is_train=True)
        test_data = MVTecDataset(c, is_train=False)
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=c.batch_size, shuffle=True, pin_memory=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, pin_memory=True
        )

    elif dataset_name == "MTD" and c.setting == "oc":
        train_data = MtdDataset(c, is_train=True)
        test_data = MtdDataset(c, is_train=False)
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=c.batch_size, shuffle=True, pin_memory=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, pin_memory=True
        )

    return train_dataloader, test_dataloader


class BaseADDataset(torch.utils.data.Dataset):
    """Base class for anomaly detection datasets to handle common transforms."""

    def __init__(self, c, is_train=True):
        self.is_train = is_train
        self.input_size = (c.image_size, c.image_size)

        # A more robust image transform that preserves aspect ratio without cropping
        self.transform_x = T.Compose(
            [
                T.Lambda(
                    lambda img: self.resize_and_pad(img, (c.image_size, c.image_size))
                ),
                T.ToTensor(),
            ]
        )
        # gt transforms should use NEAREST interpolation
        self.transform_gt = T.Compose(
            [
                T.Lambda(
                    lambda img: self.resize_and_pad(
                        img, (c.image_size, c.image_size), is_gt=True
                    )
                ),
                T.ToTensor(),
            ]
        )
        self.normalize = T.Compose(
            [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def resize_and_pad(self, img, output_size, is_gt=False):
        # Resize with aspect ratio preservation
        interpolation = Image.NEAREST if is_gt else Image.LANCZOS
        img.thumbnail(output_size, interpolation)

        # Pad to square using reflection for images and zeros for gts
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        pad_h = (output_size[1] - h) // 2
        pad_w = (output_size[0] - w) // 2
        padding = (
            (pad_h, output_size[1] - h - pad_h),
            (pad_w, output_size[0] - w - pad_w),
        )

        if is_gt:
            padded_np = np.pad(img_np, padding, mode="constant", constant_values=0)
        else:
            # For RGB images, pad each channel
            padding += ((0, 0),)
            padded_np = np.pad(img_np, padding, mode="reflect")

        return Image.fromarray(padded_np)


class MVTecDataset(BaseADDataset):
    def __init__(self, c, is_train=True, dataset="MVTecAD"):
        super().__init__(c, is_train)
        self.dataset_path = "../../../data/" + dataset
        self.class_name = c._class_
        phase = "train" if self.is_train else "test"
        self.img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        self.gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")
        # load dataset
        self.x, self.y, self.gt, _ = self.load_dataset()
        # Alias transform_gt to the common gt transform for consistency

    def __getitem__(self, idx):
        x_path, y, gt = self.x[idx], self.y[idx], self.gt[idx]
        # x = Image.open(x).convert('RGB')
        # if os.path.isfile(x):
        x = Image.open(x_path)

        if self.class_name in ["zipper", "screw", "grid"]:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)

            x = Image.fromarray(x.astype("uint8")).convert("RGB")
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            gt = torch.zeros([1, *self.input_size])
        else:
            gt = Image.open(gt)
            gt = self.transform_gt(gt)

        return x, y, gt, x_path

    def __len__(self):
        return len(self.x)

    def load_dataset(self):

        img_tot_paths = list()
        gt_tot_paths = list()
        tot_labels = list()
        tot_types = list()

        defect_types = os.listdir(self.img_dir)

        for defect_type in defect_types:
            # if self.is_vis and defect_type == "good":
            # continue
            if defect_type == "good":
                img_paths = glob.glob(os.path.join(self.img_dir, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([None] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_dir, defect_type) + "/*")
                gt_paths = glob.glob(os.path.join(self.gt_dir, defect_type) + "/*")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))

        assert len(img_tot_paths) == len(
            tot_labels
        ), "Something wrong with test and ground truth pair!"

        return img_tot_paths, tot_labels, gt_tot_paths, tot_types


class MtdDataset(BaseADDataset):
    def __init__(self, c, is_train=True, dataset="MTD"):
        super().__init__(c, is_train)  # Corrected super call
        self.dataset_path = "../../../data/" + dataset
        self.phase = "train" if is_train else "test"
        self.img_dir = os.path.join(self.dataset_path, self.phase)
        self.gt_dir = os.path.join(self.dataset_path, "ground_truth")

        # load dataset
        self.x, self.y, self.gt = self.load_dataset()

    def __getitem__(self, idx):
        x_path, y, gt_path = self.x[idx], self.y[idx], self.gt[idx]
        x = Image.open(x_path).convert("RGB")
        x = self.normalize(self.transform_x(x))

        if y == 0:
            gt = torch.zeros([1, *self.input_size])
        else:
            gt = Image.open(gt_path)
            gt = self.transform_gt(gt)

        return x, y, gt, x_path

    def __len__(self):
        return len(self.x)

    def load_dataset(self):
        img_paths = list()
        gt_paths = list()
        labels = list()

        defect_types = os.listdir(self.img_dir)

        for defect in defect_types:
            current_img_paths = glob.glob(os.path.join(self.img_dir, defect) + "/*")
            if not current_img_paths:
                continue  # Skip empty directories

            num_current_images = len(current_img_paths)
            img_paths.extend(current_img_paths)

            if defect == "good":
                gt_paths.extend([None] * num_current_images)
                labels.extend([0] * num_current_images)
            else:
                # Sort paths to ensure correspondence between images and gts
                current_img_paths.sort()
                current_gt_paths = glob.glob(os.path.join(self.gt_dir, defect) + "/*")
                current_gt_paths.sort()
                # Add an assertion for robustness
                assert len(current_img_paths) == len(current_gt_paths), (
                    f"Mismatch in number of images and gts for defect type '{defect}' in phase '{self.phase}'. "
                    f"Found {len(current_img_paths)} images and {len(current_gt_paths)} gts."
                )
                gt_paths.extend(current_gt_paths)
                labels.extend([1] * num_current_images)

        assert len(img_paths) == len(
            labels
        ), f"Number of samples do not match for {self.phase}. Images: {len(img_paths)}, Labels: {len(labels)}"

        return img_paths, labels, gt_paths
