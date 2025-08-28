from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
import cv2

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="importlib._bootstrap"
)


class ClaheTransform:
    """Applies Contrast Limited Adaptive Histogram Equalization to the L-channel of an LAB image."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_rgb)


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

    def __init__(self, c, is_train=True, dataset="MVTecAD"):
        self.is_train = is_train
        self.input_size = (c.image_size, c.image_size)

        # Image transforms that preserve aspect ratio
        self.transform_x = T.Compose(
            [
                T.Resize(c.image_size, InterpolationMode.LANCZOS),
                T.CenterCrop(c.center_crop),
                T.ToTensor(),
            ]
        )
        # Mask transforms should use NEAREST interpolation
        self.transform_gt = T.Compose(
            [
                T.Resize(c.image_size, InterpolationMode.NEAREST),
                T.CenterCrop(c.center_crop),
                T.ToTensor(),
            ]
        )
        self.normalize = T.Compose(
            [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )


class MVTecDataset(BaseADDataset):
    def __init__(self, c, is_train=True, dataset="MVTecAD"):
        super().__init__(c, is_train)
        self.dataset_path = "../../../data/" + dataset
        self.class_name = c._class_
        phase = "train" if self.is_train else "test"
        self.img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        self.gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")
        # load dataset
        self.x, self.y, self.mask, _ = self.load_dataset()

    def __getitem__(self, idx):
        x_path, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        # Use .convert('RGB') to robustly handle both color and grayscale images,
        x = Image.open(x_path).convert("RGB")
        x = self.normalize(self.transform_x(x))
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask, x_path

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
                img_paths.sort()
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


class MtdDataset(torch.utils.data.Dataset):
    def __init__(self, c, is_train=True, dataset="MTD_exp_2"):
        super().__init__()
        # Set a fixed canvas size. All images will be padded to this size.
        self.canvas_size = (c.image_size, c.image_size)
        self.dataset_path = "../../../data/" + dataset
        # Set a maximum size to resize large images, preventing OOM.
        self.max_size = c.image_size
        self.phase = "train" if is_train else "test"
        self.img_dir = os.path.join(self.dataset_path, self.phase)
        self.gt_dir = os.path.join(self.dataset_path, "ground_truth")

        # load dataset
        self.x, self.y, self.gt = self.load_dataset()

        self.normalize = T.Compose(
            [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def pad_to_canvas(self, img, canvas_size, is_mask=False):
        """Pads the image to the center of a fixed-size canvas using replication padding."""
        w, h = img.size

        pad_w = max(0, canvas_size[0] - w)
        pad_h = max(0, canvas_size[1] - h)

        # Center the image
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        padding = (left, top, right, bottom)
        padding_mode = "reflect" if not is_mask else "constant"

        return T.functional.pad(img, padding, padding_mode=padding_mode)

    def __getitem__(self, idx):
        x_path, y, gt_path = self.x[idx], self.y[idx], self.gt[idx]
        x = Image.open(x_path).convert("RGB")

        # Apply CLAHE to handle lighting inconsistencies and reflections
        x = ClaheTransform()(x)

        # Resize only if the image is larger than the max_size threshold
        w, h = x.size
        if w > self.max_size or h > self.max_size:
            # Resize while maintaining aspect ratio
            x.thumbnail((self.max_size, self.max_size), Image.LANCZOS)

        if gt_path is None:
            # For normal samples, create a black mask with the same size as the image
            gt = Image.new("L", (w, h), 0)  # Use original w, h to create the mask
        else:
            gt = Image.open(gt_path).convert("L")

        # Apply the same resize logic to the ground truth mask
        if w > self.max_size or h > self.max_size:
            gt.thumbnail((self.max_size, self.max_size), Image.NEAREST)

        # Pad both image and mask to the canvas size
        x_padded = self.pad_to_canvas(x, self.canvas_size)
        gt_padded = self.pad_to_canvas(gt, self.canvas_size, is_mask=True)

        # Convert to tensor and then normalize the image
        x_tensor = T.ToTensor()(x_padded)
        x_tensor = self.normalize(x_tensor)
        gt_tensor = T.ToTensor()(gt_padded)

        return x_tensor, y, gt_tensor, x_path

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

            current_img_paths.sort()
            num_current_images = len(current_img_paths)
            img_paths.extend(current_img_paths)

            if defect == "good":
                gt_paths.extend([None] * num_current_images)
                labels.extend([0] * num_current_images)
            else:
                current_gt_paths = glob.glob(os.path.join(self.gt_dir, defect) + "/*")
                current_gt_paths.sort()
                # Add an assertion for robustness
                assert len(current_img_paths) == len(current_gt_paths), (
                    f"Mismatch in number of images and masks for defect type '{defect}' in phase '{self.phase}'. "
                    f"Found {len(current_img_paths)} images and {len(current_gt_paths)} masks."
                )
                gt_paths.extend(current_gt_paths)
                labels.extend([1] * num_current_images)

        assert len(img_paths) == len(
            labels
        ), f"Number of samples do not match for {self.phase}. Images: {len(img_paths)}, Labels: {len(labels)}"

        return img_paths, labels, gt_paths
