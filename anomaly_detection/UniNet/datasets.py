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


# class DarkenGlare:
#     def __init__(
#         self,
#         lower=(200, 200, 200),
#         upper=(255, 255, 255),
#         strength=0.3,
#         dilate=3,
#         feather=7,
#         use_hsv_thr=False,
#     ):
#         self.lower = lower
#         self.upper = upper
#         self.strength = strength
#         self.dilate = dilate
#         self.feather = feather
#         self.use_hsv_thr = use_hsv_thr

#     def __call__(self, img):

#         if self.use_hsv_thr:
#             V = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
#             thr = max(int(np.percentile(V, 99.0)), 230)
#             mask = (V >= thr).astype(np.uint8) * 255
#         else:
#             mask = cv2.inRange(img, self.lower, self.upper)

#         if self.dilate > 0:
#             k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate, self.dilate))
#             mask = cv2.dilate(mask, k, 1)
#         m = mask.astype(np.float32) / 255.0
#         if self.feather > 0:
#             m = cv2.GaussianBlur(m, (self.feather | 1, self.feather | 1), 0)

#         # HSV에서 V만 줄이기
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         H, S, V = cv2.split(hsv)
#         Vf = V.astype(np.float32) * (1.0 - self.strength * m)
#         out = cv2.cvtColor(
#             cv2.merge([H, S, np.clip(Vf, 0, 255).astype(np.uint8)]), cv2.COLOR_HSV2BGR
#         )
#         return out

# transforms.py
import cv2, numpy as np
from PIL import Image


class DarkenGlare:
    def __init__(
        self,
        lower=(200, 200, 200),
        upper=(255, 255, 255),
        strength=0.3,
        dilate=3,
        feather=7,
        use_hsv_thr=False,
        v_pct=99.0,
        v_abs=230,
        s_max=80,
    ):
        self.lower, self.upper = lower, upper
        self.strength, self.dilate, self.feather = strength, dilate, feather
        self.use_hsv_thr, self.v_pct, self.v_abs, self.s_max = (
            use_hsv_thr,
            v_pct,
            v_abs,
            s_max,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        # PIL -> ndarray(BGR)
        if img.mode == "RGB":
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            alpha = None
        elif img.mode == "RGBA":
            rgba = np.asarray(img)
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            alpha = rgba[..., 3]
        elif img.mode == "L":
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_GRAY2BGR)
            alpha = None
        else:
            bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            alpha = None

        # 마스크
        if self.use_hsv_thr:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            V, S = hsv[..., 2], hsv[..., 1]
            thr = max(int(np.percentile(V, self.v_pct)), self.v_abs)
            mask = ((V >= thr) & (S <= self.s_max)).astype(np.uint8) * 255
        else:
            mask = cv2.inRange(bgr, self.lower, self.upper)

        if self.dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate, self.dilate))
            mask = cv2.dilate(mask, k, 1)
        m = mask.astype(np.float32) / 255.0
        if self.feather > 0:
            ksz = self.feather | 1
            m = cv2.GaussianBlur(m, (ksz, ksz), 0)

        # HSV V만 감산
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        Vf = np.clip(V.astype(np.float32) * (1.0 - self.strength * m), 0, 255).astype(
            np.uint8
        )
        out_bgr = cv2.cvtColor(cv2.merge([H, S, Vf]), cv2.COLOR_HSV2BGR)

        # ndarray(BGR) -> PIL
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        if alpha is not None:
            out = Image.fromarray(np.dstack([out_rgb, alpha]), mode="RGBA")
        else:
            out = Image.fromarray(out_rgb, mode="RGB")
        return out


class BaseADDataset(torch.utils.data.Dataset):
    """Base class for anomaly detection datasets to handle common transforms."""

    def __init__(self, c, is_train=True, dataset="MVTecAD"):
        self.is_train = is_train
        self.input_size = (c.image_size, c.image_size)

        # Image transforms that preserve aspect ratio
        self.transform_x = T.Compose(
            [
                T.Resize(c.image_size, InterpolationMode.LANCZOS),
                DarkenGlare(),
                T.ToTensor(),
            ]
        )
        # Mask transforms should use NEAREST interpolation
        self.transform_gt = T.Compose(
            [
                T.Resize(c.image_size, InterpolationMode.NEAREST),
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
    def __init__(self, c, is_train=True, dataset="MTD_exp"):
        super().__init__()
        self.image_size = (c.image_size, c.image_size)
        self.dataset_path = "../../../data/" + dataset
        self.phase = "train" if is_train else "test"
        self.img_dir = os.path.join(self.dataset_path, self.phase)
        self.gt_dir = os.path.join(self.dataset_path, "ground_truth")

        # load dataset
        self.x, self.y, self.gt = self.load_dataset()

    def __getitem__(self, idx):
        x_path, y, gt_path = self.x[idx], self.y[idx], self.gt[idx]
        x = Image.open(x_path).convert("RGB")

        # Transforms
        self.transform_x = T.Compose(
            [
                T.Resize(self.image_size, InterpolationMode.LANCZOS),
                DarkenGlare(),
                T.ToTensor(),
            ]
        )

        self.transform_gt = T.Compose(
            [
                T.Resize(self.image_size, InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )
        self.normalize = T.Compose(
            [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        x = self.normalize(self.transform_x(x))

        if y == 0:
            gt = torch.zeros([1, *self.image_size])
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
