import math
import re
import time
import cv2
import os  # Added for path manipulation
import torch
import numpy as np
from skimage.measure import regionprops
from torch.nn import functional as F
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.metrics import auc

from utils import t2np, rescale
from functools import partial
from multiprocessing import Pool
from skimage.measure import label, regionprops
from PIL import Image
from torchvision import transforms as T
from visualization import (
    save_anomaly_visualization,
)  # Import the visualization function

from UniNet_lib.mechanism import weighted_decision_mechanism


def tiling_inference(model, image_path, device, image_size=256, stride_ratio=0.5):
    """
    Performs tiling inference on a single high-resolution image.
    Returns a full-resolution anomaly map.
    """
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape

    full_anomaly_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    stride = int(image_size * stride_ratio)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # To handle edges, always take a full-sized patch
            y_end = min(y + image_size, h)
            x_end = min(x + image_size, w)
            y_start = max(0, y_end - image_size)
            x_start = max(0, x_end - image_size)
            patch_np = img_np[y_start:y_end, x_start:x_end]

            patch_pil = Image.fromarray(patch_np)
            patch_tensor = T.ToTensor()(patch_pil)
            patch_tensor = normalize(patch_tensor).unsqueeze(0).to(device)

            # The model's forward pass for evaluation returns feature maps
            t_tf, de_features = model(
                patch_tensor
            )  # t_tf: Teacher features, de_features: Student features

            # Calculate anomaly score from features for the patch by interpolating to a common size
            # and then computing cosine similarity.
            patch_anomaly_score_components = []
            for t, s in zip(t_tf, de_features):
                # Interpolate both teacher and student features to the model's input image_size
                # This ensures spatial dimensions match for cosine_similarity
                t_resized = F.interpolate(
                    t,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                s_resized = F.interpolate(
                    s,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )

                patch_anomaly_score_components.append(
                    1 - F.cosine_similarity(t_resized, s_resized, dim=1)
                )

            # Sum the anomaly scores from different feature levels
            patch_anomaly_score_tensor = sum(patch_anomaly_score_components)

            # The result is already at the target size, so just convert to numpy
            patch_anomaly_score = (
                patch_anomaly_score_tensor.squeeze().detach().cpu().numpy()
            )

            # Get the actual height and width of the patch to crop the score map
            patch_h = y_end - y_start
            patch_w = x_end - x_start

            # Add the relevant part of the score map to the full anomaly map
            full_anomaly_map[y_start:y_end, x_start:x_end] += patch_anomaly_score[
                :patch_h, :patch_w
            ]
            count_map[y_start:y_end, x_start:x_end] += 1

    return full_anomaly_map / (count_map + 1e-8)


def evaluation_indusAD(
    c, model, dataloader, device, save_visuals=False
):  # Changed is_train to save_visuals
    model.train_or_eval(type="eval")
    n = model.n

    # Get the full list of test images and ground truths from the dataset object
    test_image_paths = dataloader.dataset.x
    test_gt_paths = dataloader.dataset.gt
    test_labels = dataloader.dataset.y

    all_anomaly_scores = []  # For image-level metrics
    all_pixel_scores = []  # For full-resolution P-AUROC
    all_pixel_labels = []  # For full-resolution P-AUROC
    all_anomaly_maps_for_pro = []  # For resized PRO calculation
    all_gt_masks_for_pro = []  # For resized PRO calculation

    start_time = time.time()
    with torch.no_grad():
        for i, image_path in enumerate(test_image_paths):
            # Perform tiling inference to get a full-resolution anomaly map
            full_res_anomaly_map = tiling_inference(
                model, image_path, device, image_size=c.image_size
            )

            # Calculate image-level score (e.g., max value of the map)
            image_score = np.max(full_res_anomaly_map)
            all_anomaly_scores.append(image_score)

            # For P-AUROC: use full-resolution maps
            gt_path = test_gt_paths[i]
            if gt_path is None:
                # For 'good' samples, create a zero mask of the same size
                full_res_gt_mask = np.zeros_like(full_res_anomaly_map)
            else:
                # For 'bad' samples, load the GT mask and resize it to match the anomaly map's full resolution
                full_res_gt_mask = np.array(
                    Image.open(gt_path).resize(
                        full_res_anomaly_map.T.shape, Image.NEAREST
                    )
                )
            all_pixel_scores.append(full_res_anomaly_map.flatten())
            all_pixel_labels.append(full_res_gt_mask.flatten())

            # For PRO metric: use resized maps as a pragmatic approach for the existing eval_seg_pro function
            resized_map_for_pro = cv2.resize(
                full_res_anomaly_map, (c.image_size, c.image_size)
            )
            all_anomaly_maps_for_pro.append(resized_map_for_pro)
            resized_gt_mask_for_pro = cv2.resize(
                full_res_gt_mask,
                (c.image_size, c.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            all_gt_masks_for_pro.append(resized_gt_mask_for_pro)

            # Save visualization if requested for abnormal samples
            if save_visuals and test_labels[i] == 1:
                original_image_tensor = T.ToTensor()(
                    Image.open(image_path).convert("RGB")
                )
                save_anomaly_visualization(
                    original_image=original_image_tensor,
                    anomaly_map=full_res_anomaly_map,  # Use the full-res map for visualization
                    save_path=os.path.join(
                        c.save_dir,
                        "visuals",
                        c.dataset,
                        c._class_,
                        os.path.basename(image_path),
                    ),
                )

    fps = len(test_image_paths) / (time.time() - start_time)
    print("fps:", fps, len(test_image_paths))

    # Convert lists to numpy arrays for metric calculation
    # Image-level metrics
    anomaly_scores = np.array(all_anomaly_scores)
    gt_labels = np.array(test_labels, dtype=np.bool_)
    auroc_sp = round(roc_auc_score(gt_labels, anomaly_scores) * 100, 1)
    ap = round(average_precision_score(gt_labels, anomaly_scores) * 100, 1)

    # Pixel-level AUROC (full resolution)
    final_pixel_scores = np.concatenate(all_pixel_scores)
    final_pixel_labels = np.concatenate(all_pixel_labels).astype(np.bool_)
    auroc_px = round(roc_auc_score(final_pixel_labels, final_pixel_scores) * 100, 1)

    # PRO (resized)
    anomaly_maps_for_pro = np.array(all_anomaly_maps_for_pro)
    gt_masks_for_pro = np.array(all_gt_masks_for_pro, dtype=np.bool_)
    pro = round(eval_seg_pro(gt_masks_for_pro, anomaly_maps_for_pro), 1)

    return auroc_px, auroc_sp, pro, ap


def eval_seg_pro(gt_mask, anomaly_score_map, max_step=800):
    expect_fpr = 0.3  # default 30%

    max_th = anomaly_score_map.max()
    min_th = anomaly_score_map.min()
    delta = (max_th - min_th) / max_step
    threds = np.arange(min_th, max_th, delta).tolist()

    pool = Pool(8)
    ret = pool.map(partial(single_process, anomaly_score_map, gt_mask), threds)
    pool.close()
    pros_mean = []
    fprs = []
    for pro_mean, fpr in ret:
        pros_mean.append(pro_mean)
        fprs.append(fpr)
    pros_mean = np.array(pros_mean)
    fprs = np.array(fprs)
    # expect_fpr = sum(fprs) / len(fprs)
    idx = (
        fprs < expect_fpr
    )  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    loc_pro_auc = auc(fprs_selected, pros_mean_selected) * 100

    return loc_pro_auc


def single_process(anomaly_score_map, gt_mask, thred):
    binary_score_maps = np.zeros_like(anomaly_score_map, dtype=np.bool_)
    binary_score_maps[anomaly_score_map <= thred] = 0
    binary_score_maps[anomaly_score_map > thred] = 1
    pro = []
    for binary_map, mask in zip(binary_score_maps, gt_mask):  # for i th image
        for region in regionprops(label(mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            tp_pixels = binary_map[axes0_ids, axes1_ids].sum()
            pro.append(tp_pixels / region.area)

    pros_mean = np.array(pro).mean()
    inverse_masks = 1 - gt_mask
    fpr = np.logical_and(inverse_masks, binary_score_maps).sum() / inverse_masks.sum()
    return pros_mean, fpr


def evaluation_batch(
    c, model, dataloader, device, _class_=None, reg_calib=False, max_ratio=0
):
    model.train_or_eval(type="eval")
    gt_list_sp = []
    output_list = [list() for i in range(6)]
    weights_cnt = 0

    with torch.no_grad():
        for img, gt, label, cls in dataloader:
            img = img.to(device)
            gt_list_sp.extend(t2np(label))
            t_tf, de_features = model(img)
            weights_cnt += 1

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)  # B*64*64
                # print(output, output.shape)
                output_list[l].append(output)

        anomaly_score, _ = weighted_decision_mechanism(
            weights_cnt, output_list, c.alpha, c.beta
        )

        # anomaly_score = gaussian_filter(anomaly_score, sigma=4)
        gt_list_sp = np.asarray(gt_list_sp, dtype=np.bool_)
        # pr_list_sp.extend(sp_score)

        auroc_sp = round(roc_auc_score(gt_list_sp, anomaly_score), 4)
        ap_sp = round(average_precision_score(gt_list_sp, anomaly_score), 4)
        f1_sp = f1_score_max(gt_list_sp, anomaly_score)

    return auroc_sp, ap_sp, f1_sp


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def extract_numbers(file_name):
    numbers = re.findall(r"(\d+)", file_name)
    return tuple(map(int, numbers))


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return 1.0 - ((psnr - min_psnr) / (max_psnr - min_psnr + 1e-8))


def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    max_ele = np.max(psnr_list)
    min_ele = np.min(psnr_list)
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], max_ele, min_ele))

    return anomaly_score_list
