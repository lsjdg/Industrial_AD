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


def loading_dataset(c, dataset_name):
    train_dataloader, test_dataloader = None, None

    if dataset_name == "MTD" and c.setting == "oc":
        train_data = MtdDataset(c, is_train=True)
        test_data = MtdDataset(c, is_train=False)
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=c.batch_size, shuffle=True, pin_memory=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, pin_memory=True
        )

    return train_dataloader, test_dataloader


class ClaheTransform:
    """Applies Contrast Limited Adaptive Histogram Equalization to the L-channel of an LAB image."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img.convert("RGB"))
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_rgb)


class UnifiedTransform:
    """
    A unified transform pipeline that handles conditional resizing, padding, and CLAHE.
    This ensures consistency between training and evaluation.
    """

    def __init__(self, c, use_clahe=False):
        self.canvas_size = (c.image_size, c.image_size)
        self.max_size = c.image_size
        self.use_clahe = use_clahe

        if self.use_clahe:
            self.clahe = ClaheTransform()

        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, img, mask=None):
        if self.use_clahe:
            img = self.clahe(img)

        w, h = img.size
        if w > self.max_size or h > self.max_size:
            img.thumbnail((self.max_size, self.max_size), Image.LANCZOS)

        if mask is None:
            mask = Image.new("L", img.size, 0)
        else:
            mask.thumbnail(img.size, Image.NEAREST)

        img_padded = self._pad_to_canvas(img, self.canvas_size, is_mask=False)
        mask_padded = self._pad_to_canvas(mask, self.canvas_size, is_mask=True)

        img_tensor = T.ToTensor()(img_padded)
        mask_tensor = T.ToTensor()(mask_padded)

        img_tensor = self.normalize(img_tensor)

        return img_tensor, mask_tensor

    def _pad_to_canvas(self, img, canvas_size, is_mask=False):
        """Pads the image to the center of a fixed-size canvas."""
        w, h = img.size
        pad_w = max(0, canvas_size[0] - w)
        pad_h = max(0, canvas_size[1] - h)

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        padding = (left, top, right, bottom)
        padding_mode = "reflect" if not is_mask else "constant"
        fill = 0

        return T.functional.pad(
            img,
            padding,
            fill=fill,
            padding_mode=padding_mode,
        )


class MtdDataset(torch.utils.data.Dataset):
    def __init__(self, c, is_train=True, dataset="MTD_exp"):
        super().__init__()
        self.dataset_path = "../../../data/" + dataset
        self.phase = "train" if is_train else "test"
        self.img_dir = os.path.join(self.dataset_path, self.phase)
        self.gt_dir = os.path.join(self.dataset_path, "ground_truth")

        # Use the unified transform for both training and testing
        self.transform = UnifiedTransform(c, use_clahe=False)

        # load dataset
        self.x, self.y, self.gt = self.load_dataset()

    def __getitem__(self, idx):
        x_path, y, gt_path = self.x[idx], self.y[idx], self.gt[idx]
        x = Image.open(x_path).convert("RGB")

        gt = Image.open(gt_path).convert("L") if gt_path is not None else None

        x_tensor, gt_tensor = self.transform(x, gt)

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
