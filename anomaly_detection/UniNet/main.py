import numpy as np
import os

from train_unsupervisedAD import train
from datasets import (
    unsupervised,
    mvtec_list,
    industrial,
)
import argparse
from utils import setup_seed, get_logger
from test import test


def parsing_args():
    parser = argparse.ArgumentParser(description="UniNet")

    parser.add_argument(
        "--domain",
        default="industrial",
        type=str,
        choices=["industrial"],
        help="choose experimental domain.",
    )
    parser.add_argument(
        "--setting",
        default="oc",
        type=str,
        choices=["oc", "mc", "cd"],
        help="choose experimental settings, including one-class, multi-class, cross-dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="MVTecAD",
        type=str,
        choices=[
            "MVTecAD",
        ],
        help="choose experimental dataset.",
    )
    parser.add_argument("--epochs", default=100, type=int, help="epochs.")
    parser.add_argument("--batch_size", default=8, type=int, help="batch sizes.")
    parser.add_argument("--image_size", default=256, type=int, help="image size.")
    parser.add_argument("--center_crop", default=256, type=int, help="crop image size.")
    parser.add_argument(
        "--lr_s", default=5e-3, type=float, help="lr for student."
    )  # 5e-3
    parser.add_argument(
        "--lr_t", default=1e-6, type=float, help="lr for teacher."
    )  # 1e-6
    parser.add_argument(
        "--T", default=2, type=float, help="temperature for contrastive learning."
    )

    parser.add_argument(
        "--weighted_decision_mechanism",
        action="store_true",
        default=True,
        help="whether to employ weight-guided similarity to calculate anomaly map.",
    )
    parser.add_argument(
        "--default", default=0.3, type=float, help="the default value of weights."
    )
    parser.add_argument(
        "--alpha", default=0.01, type=float, help="hyperparameters for weights."
    )
    parser.add_argument(
        "--beta", default=0.00003, type=float, help="hyperparameters for weights."
    )

    parser.add_argument(
        "--train_and_test_all",
        action="store_true",
        default=False,
        help="for medical domains.",
    )
    parser.add_argument(
        "--is_saved",
        action="store_true",
        default=True,
        help="whether to save model weights.",
    )
    parser.add_argument("--save_dir", type=str, default="./saved_results")
    parser.add_argument(
        "--load_ckpts",
        action="store_true",
        default=False,
        help="loading ckpts for testing",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    setup_seed(1203)
    c = parsing_args()
    if not c.weighted_decision_mechanism:
        c.default = c.alpha = c.beta = c.gamma = "w/o"

    dataset_name = c.dataset
    logger = get_logger(dataset_name, os.path.join(c.save_dir, dataset_name))

    dataset = None
    if dataset_name in industrial:
        c.domain = "industrial"
        if dataset_name == "MVTecAD":
            dataset = mvtec_list

    else:
        raise KeyError(f"Dataset '{dataset_name}' can not be found.")

    from tabulate import tabulate

    results = {}
    table_ls = []

    # ---------------------------------------------------------------------------------------------------------
    # --------------------------------------unsupervised industrial AD-----------------------------------------
    # ---------------------------------------------------------------------------------------------------------
    if dataset_name in industrial and dataset_name in unsupervised:
        image_auroc_list = []
        pixel_auroc_list = []
        pixel_aupro_list = []

        # -----------------------------train-------------------------------------
        if not c.load_ckpts:
            for idx, i in enumerate(dataset):
                c._class_ = i

                args_dict = vars(c)
                args_info = f"class:{i}, " if c.setting == "oc" else f""
                for key, value in args_dict.items():
                    if key in ["_class_"]:
                        continue
                    args_info += ", ".join([f"{key}:{value}, "])

                if c.setting == "oc":
                    (
                        print(
                            "training on {} dataset (separate-class)".format(
                                dataset_name
                            )
                        )
                        if idx == 0
                        else None
                    )
                    print(args_info)
                    train(c)

            print("training over!")

        # -----------------------------test-------------------------------------
        if c.setting == "oc":
            for idx, i in enumerate(dataset):
                (
                    print(f"testing on {dataset_name} dataset (separate-class)")
                    if idx == 0
                    else None
                )
                c._class_ = i
                print(f"testing class:{i}")
                auroc_sp, auroc_px, aupro_px = test(c, suffix="BEST_P_PRO")
                print("")
                table_ls.append(
                    [
                        "{}".format(i),
                        str(np.round(auroc_sp, decimals=1)),
                        str(np.round(auroc_px, decimals=1)),
                        str(np.round(aupro_px, decimals=1)),
                    ]
                )
                image_auroc_list.append(auroc_sp)
                pixel_auroc_list.append(auroc_px)
                pixel_aupro_list.append(aupro_px)
                results = tabulate(
                    table_ls,
                    headers=["object", "image_auroc", "pixel_auroc", "pixel_aupro"],
                    tablefmt="pipe",
                )
            table_ls.append(
                [
                    "mean",
                    str(np.round(np.mean(image_auroc_list), decimals=2)),
                    str(np.round(np.mean(pixel_auroc_list), decimals=2)),
                    str(np.round(np.mean(pixel_aupro_list), decimals=2)),
                ]
            )
            results = tabulate(
                table_ls,
                headers=["object", "image_auroc", "pixel_auroc", "pixel_aupro"],
                tablefmt="pipe",
            )
            print(results)
