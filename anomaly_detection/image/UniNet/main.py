import argparse


def parsing_args():
    parser = argparse.ArgumentParser(description="UniNet")

    parser.add_argument(
        "--setting",
        default="oc",
        type=str,
        choices=["oc", "mc", "cd"],
        help="choose experimental settings among [one-class (oc), multi-class(mc), cross-dataset(cd)].",
    )
    parser.add_argument(
        "--dataset",
        default="MVTecAD",
        type=str,
        choices=["MVTecAD", "MTD", "DAC"],
        help="choose dataset.",
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
        "--disable-wdm",
        action="store_false",
        dest="wdm",  # weight decision mechanism
        help="Disable the weight-guided similarity for anomaly map calculation (it is enabled by default).",
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
