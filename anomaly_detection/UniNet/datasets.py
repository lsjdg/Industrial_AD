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


# BTAD_PATH = os.path.abspath(os.path.join("D:\ws/btad"))


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

    return train_dataloader, test_dataloader


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, c, is_train=True, dataset="MVTecAD"):
        self.dataset_path = "../../../data/" + dataset
        self.class_name = c._class_
        self.is_train = is_train

        self.input_size = (c.image_size, c.image_size)
        self.aug = False
        phase = "train" if self.is_train else "test"
        self.img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        self.gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")
        # load dataset
        self.x, self.y, self.mask, _ = self.load_dataset()
        # set transforms
        if is_train:
            self.transform_x = T.Compose(
                [T.Resize(self.input_size, InterpolationMode.LANCZOS), T.ToTensor()]
            )
        # test:
        else:
            self.transform_x = T.Compose(
                [T.Resize(self.input_size, InterpolationMode.LANCZOS), T.ToTensor()]
            )
        # mask
        self.transform_mask = T.Compose(
            [T.Resize(self.input_size, InterpolationMode.NEAREST), T.ToTensor()]
        )

        self.normalize = T.Compose(
            [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def __getitem__(self, idx):
        x_path, y, mask = self.x[idx], self.y[idx], self.mask[idx]
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
    def __init__(self, c, is_train=True, dataset="MTD"):
        self.dataset_path = "../../../data/" + dataset
        self.phase = "train" if is_train else "test"

        self.input_size = (c.image_size, c.image_size)
        self.img_dir = os.path.join(self.dataset_path, self.phase)
        self.gt_dir = os.path.join(self.dataset_path, "ground_truth")

        # load dataset
        self.x, self.y, self.gt = self.load_dataset()

        # transforms
        self.transform_x = T.Compose(
            [T.Resize(self.input_size, InterpolationMode.LANCZOS), T.ToTensor()]
        )
        self.transform_gt = T.Compose(
            T.Resize(self.input_size, InterpolationMode.NEAREST), T.ToTensor()
        )
        self.normalize = T.Compose(
            [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

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
            if defect == "good":
                img_paths.extend(os.path.join(self.img_dir, defect) + "/*")
                gt_paths.extend([None] * len(img_paths))
                labels.extend([0] * len(img_paths))
            else:
                img_paths.extend(os.path.join(self.img_dir, defect) + "/*")
                gt_paths.extend(os.path.join(self.gt_dir, defect) + "/*")
                labels.extend([1] * len(img_paths))

        assert len(img_paths) != len(labels), "Number of samples do not match"

        return img_paths, labels, gt_paths
