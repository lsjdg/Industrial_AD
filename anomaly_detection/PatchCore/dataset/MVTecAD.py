import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

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
        path="../datasets/MVTecAD",
        class_name="bottle",
        is_train=True,  # True if trian set
        resize=256,
        crop=224,
    ):
        # check availability of class name
        assert class_name in CLASSES, f"class {class_name} is not available."

        self.path = path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.crop = crop

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose(
            [
                T.Resize(resize, interpolation=Image.BICUBIC),
                T.CenterCrop(crop),
                T.ToTensor(),
                # Normalization status of ImageNet pretrained models
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Nearest Neighbor Interpolation is used for masks
        self.transform_mask = T.Compose(
            [T.Resize(resize, Image.NEAREST), T.CenterCrop(crop), T.ToTensor()]
        )

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert("RGB")
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.crop, self.crop])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        x, y, mask = [], [], []

        image_dir = os.path.join(self.path, self.class_name, phase)
        gt_dir = os.path.join(
            self.path, self.class_name, "ground_truth"
        )  # ground truth 폴더

        img_types = sorted(os.listdir(image_dir))  # good, broken_large ...
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(image_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            image_filepaths = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".png")
                ]
            )
            x.extend(image_filepaths)

            # load ground truth labels
            if img_type == "good":
                y.extend([0] * len(image_filepaths))
                mask.extend([None] * len(image_filepaths))
            else:
                y.extend([1] * len(image_filepaths))
                gt_type_dir = os.path.join(gt_dir, img_type)
                # splitext: basename -> (filename, extension)
                img_fname_list = [
                    os.path.splitext(os.path.basename(f))[0] for f in image_filepaths
                ]
                # Construct mask filepaths based on image filenames
                # Expects masks to be named like image_mask.png in the ground_truth directory

                gt_fpath_list = [
                    os.path.join(gt_type_dir, img_fname + "_mask.png")
                    for img_fname in img_fname_list
                ]
                mask.extend(gt_fpath_list)

            assert len(x) == len(y), "x and y should be the same length"

        return list(x), list(y), list(mask)
