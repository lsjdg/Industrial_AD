import torch
import numpy as np
import cv2
import os


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def save_anomaly_visualization(
    original_image: torch.Tensor,
    anomaly_map: np.ndarray,
    save_path: str = None,
    alpha: float = 0.5,
):
    """Creates and saves a visualization of the anomaly map overlaid on the original image."""
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Prepare the original image
    if original_image.dim() == 4:
        original_image = original_image.squeeze(0)
    img_np = denormalize(original_image.cpu()).permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2. Prepare the anomaly map
    heatmap_norm = cv2.normalize(
        anomaly_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    heatmap_color = cv2.applyColorMap(
        heatmap_norm, cv2.COLORMAP_HOT
    )  # HOT colormap gives yellow-ish feel

    # 3. Blend the original image with the heatmap
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)

    if save_path is not None:
        cv2.imwrite(save_path, superimposed_img)

    return superimposed_img
