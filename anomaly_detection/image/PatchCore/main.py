import os, torch, argparse
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models.PatchCore import PatchCore
from dataset.MVTecAD import MVTecDataset

from multiprocessing import freeze_support
from models.backbones import *

# Using a dictionary to map names to functions is safer than eval()
supported_backbones = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "wideresnet50": wideresnet50,
}


def main(class_name, backbone, backbone_name, device="cuda"):
    BS_TRAIN, BS_TEST = 16, 1
    MEM_BANK_PATH = f"./memory_bank/{backbone_name}"
    MEM_BANK_FILE = f"{MEM_BANK_PATH}/{class_name}.pth"
    os.makedirs(MEM_BANK_PATH, exist_ok=True)
    train_ds = MVTecDataset(class_name=class_name, is_train=True)
    test_ds = MVTecDataset(class_name=class_name, is_train=False)
    train_loader = DataLoader(
        train_ds, batch_size=BS_TRAIN, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(test_ds, batch_size=BS_TEST, shuffle=False, num_workers=4)

    model = PatchCore(backbone=backbone, memory_bank_path=MEM_BANK_PATH, device=device)
    model.backbone.eval()

    print("▶ Checking for existing memory bank…")

    if not os.path.isfile(MEM_BANK_FILE):
        print("▶ Building memory bank…")
        all_features = []
        with torch.no_grad():
            for imgs, _, _ in train_loader:
                imgs = imgs.to(device)
                features = model.extract_features(imgs)
                all_features.append(features.cpu())
        all_features = torch.cat(all_features, dim=0)
        model.build_memory_bank(all_features, MEM_BANK_FILE)

    print("▶ Loading memory bank…")
    model.load_memory(MEM_BANK_FILE)

    print("▶ Inference on test set…")
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, label, _ in test_loader:
            imgs = imgs.to(device)  # Move input to the correct device
            batch_scores = model(imgs)
            scores.append(batch_scores.item())
            labels.append(int(label))

    auroc = roc_auc_score(labels, scores)
    print(f"[{class_name}] Image-level AUROC: {auroc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=str,
        default="wideresnet50",
        choices=supported_backbones.keys(),
        help="Name of the backbone CNN to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu).",
    )
    args = parser.parse_args()

    freeze_support()

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

    backbone_name = args.backbone

    if backbone_name not in supported_backbones:
        raise ValueError(f"Backbone {backbone_name} not supported.")
    backbone = supported_backbones[backbone_name]

    print("+" * 100)
    print("Experiment Settings:")
    print(f"device: {args.device}")
    print(f"backbone: {backbone_name}")

    for c in CLASSES:
        print("+" * 100)
        print(f"Class: {c}")
        print("+" * 100)
        main(c, backbone, backbone_name, device=args.device)
