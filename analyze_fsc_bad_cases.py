import argparse
import csv
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GDDINO_DIR = os.path.join(ROOT_DIR, "GroundingDINO")
if GDDINO_DIR not in sys.path:
    sys.path.insert(0, GDDINO_DIR)

from groundingdino.util.base_api import load_model, threshold_box
from utils.image_loader_fsc147 import get_fsc_loader
from utils.tester_fsc import (
    _build_overlap_grid,
    _dedup_boxes_by_center,
    calc_loc_metric,
    prepare_targets,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py",
        type=str,
    )
    parser.add_argument("--pretrain_model", required=True, type=str)
    parser.add_argument("--load_mode", default="test", choices=["val", "test"], type=str)
    parser.add_argument("--eval_impl", default="eval_reproduce", choices=["eval_fn", "eval_reproduce"], type=str)
    parser.add_argument("--stats_dir", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--threshold", default=0.35, type=float)
    parser.add_argument("--pred_num_judge", default=500, type=int)
    parser.add_argument("--scale", default=1000, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--seed", default=314, type=int)
    parser.add_argument("--topk_vis", default=20, type=int)
    parser.add_argument("--dense_threshold", default=0.25, type=float)
    parser.add_argument("--split_rows", default=3, type=int)
    parser.add_argument("--split_cols", default=3, type=int)
    parser.add_argument("--split_overlap", default=0.1, type=float)
    parser.add_argument("--dedup_radius_px", default=6.0, type=float)
    parser.add_argument("--horizon_flip", action="store_true")
    parser.add_argument("--horizon_flip_prob", default=0.5, type=float)
    parser.add_argument("--vertical_flip", action="store_false")
    parser.add_argument("--vertical_flip_prob", default=0.5, type=float)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def unnormalize_image(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    return img


def remap_boxes_from_crop(boxes, top, left, crop_h, crop_w, full_h, full_w):
    boxes = boxes.clone()
    if boxes.numel() == 0:
        return boxes
    boxes[:, 0] = (boxes[:, 0] * crop_w + left) / full_w
    boxes[:, 1] = (boxes[:, 1] * crop_h + top) / full_h
    boxes[:, 2] = boxes[:, 2] * crop_w / full_w
    boxes[:, 3] = boxes[:, 3] * crop_h / full_h
    return boxes


def run_eval_reproduce_single(model, images, texts, outputs, args):
    results = threshold_box(outputs, args.threshold)
    pred_num_judge = len(results[0][1])
    split_used = False
    active_threshold = args.dense_threshold if pred_num_judge > args.pred_num_judge else args.threshold
    if active_threshold != args.threshold:
        results = threshold_box(outputs, active_threshold)
        pred_num_judge = len(results[0][1])

    if pred_num_judge <= args.pred_num_judge:
        return results, split_used, pred_num_judge

    split_used = True
    h, w = images.shape[-2:]
    resize_transform = T.Resize(800)
    crop_specs = _build_overlap_grid(h, w, args.split_rows, args.split_cols, args.split_overlap)

    remapped_boxes_list = []
    logits_list = []
    for top, bottom, left, right in crop_specs:
        crop_h = bottom - top
        crop_w = right - left
        cropped = TF.crop(images[0], top, left, crop_h, crop_w)
        resized_crop = resize_transform(cropped).unsqueeze(0)
        with torch.no_grad():
            crop_outputs = model(resized_crop, captions=texts)
            crop_results = threshold_box(crop_outputs, threshold=active_threshold)
        crop_boxes, crop_logits = crop_results[0]
        remapped_boxes = remap_boxes_from_crop(crop_boxes, top, left, crop_h, crop_w, h, w)
        remapped_boxes_list.append(remapped_boxes)
        logits_list.append(crop_logits)

    final_boxes = torch.cat(remapped_boxes_list, dim=0)
    final_logits = torch.cat(logits_list, dim=0)
    final_boxes, final_logits = _dedup_boxes_by_center(
        final_boxes,
        final_logits,
        h,
        w,
        args.dedup_radius_px,
    )
    return [(final_boxes, final_logits)], split_used, pred_num_judge


def evaluate_one_sample(model, batch, args):
    images, patches, targets, boxes, texts, file_name = batch
    density_maps = targets["density_map"].squeeze(1).to(device)
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images, captions=texts)

    pred_density_map = outputs["density"].squeeze(1)
    pred_reg_count = float(torch.sum(pred_density_map, dim=[1, 2])[0].item() / args.scale)

    outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]
    h, w = images.shape[-2:]
    emb_size = outputs["pred_logits"].shape[2]
    prepared_targets = prepare_targets(targets, emb_size, [h, w])

    if args.eval_impl == "eval_reproduce":
        final_results, split_used, init_query_count = run_eval_reproduce_single(model, images, texts, outputs, args)
    else:
        final_results = threshold_box(outputs, args.threshold)
        split_used = False
        init_query_count = len(final_results[0][1])

    pred_boxes_tensor, pred_logits_tensor = final_results[0]
    pred_boxes_list = [box.tolist() for box in pred_boxes_tensor]
    pred_points = np.array([[box[0], box[1]] for box in pred_boxes_list], dtype=np.float32)
    if pred_points.size == 0:
        pred_points = np.zeros((0, 2), dtype=np.float32)

    gt_points = prepared_targets[0]["points"].detach().cpu().numpy()
    gt_count = int(gt_points.shape[0])
    det_count = int(len(pred_boxes_list))
    det_abs_err = abs(det_count - gt_count)
    reg_abs_err = abs(pred_reg_count - gt_count)
    tp, fp, fn, precision, recall, f1 = calc_loc_metric(pred_boxes_list, prepared_targets[0]["points"])

    return {
        "file_name": file_name[0],
        "text": texts[0],
        "gt_count": gt_count,
        "det_count": det_count,
        "reg_count": pred_reg_count,
        "det_abs_err": det_abs_err,
        "reg_abs_err": reg_abs_err,
        "det_minus_gt": det_count - gt_count,
        "reg_minus_gt": pred_reg_count - gt_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "split_used": int(split_used),
        "init_query_count": int(init_query_count),
        "image_tensor": images[0].detach().cpu(),
        "pred_density_map": pred_density_map[0].detach().cpu(),
        "gt_points": gt_points,
        "pred_points": pred_points,
    }


def save_separate_visuals(record, points_save_path, density_save_path):
    image = unnormalize_image(record["image_tensor"])
    pred_density = record["pred_density_map"].numpy()
    pred_density = (pred_density - pred_density.min()) / (pred_density.max() - pred_density.min() + 1e-8)

    h, w = image.shape[:2]
    gt_points = record["gt_points"] * np.array([w, h], dtype=np.float32)
    pred_points = record["pred_points"] * np.array([w, h], dtype=np.float32)

    # 点图：红色星标 Pred 中心，绿色点标 GT 中心
    fig_points, ax_points = plt.subplots(1, 1, figsize=(8, 6), dpi=220)
    ax_points.imshow(image)
    if len(gt_points) > 0:
        ax_points.scatter(
            gt_points[:, 0],
            gt_points[:, 1],
            c="lime",
            s=16,
            marker="o",
            edgecolors="black",
            linewidths=0.35,
        )
    if len(pred_points) > 0:
        ax_points.scatter(
            pred_points[:, 0],
            pred_points[:, 1],
            c="red",
            s=30,
            marker="*",
            edgecolors="black",
            linewidths=0.35,
        )
    ax_points.axis("off")
    fig_points.tight_layout(pad=0.0)
    fig_points.savefig(points_save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_points)

    # 回归密度图：单独输出热力图，便于后续拼接
    fig_density, ax_density = plt.subplots(1, 1, figsize=(8, 6), dpi=220)
    ax_density.imshow(pred_density, cmap="magma")
    ax_density.axis("off")
    fig_density.tight_layout(pad=0.0)
    fig_density.savefig(density_save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_density)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir or os.path.join(args.stats_dir, f"badcase_{args.load_mode}_{args.eval_impl}")
    points_vis_dir = os.path.join(output_dir, "point_maps")
    density_vis_dir = os.path.join(output_dir, "reg_density_maps")
    case_record_dir = os.path.join(output_dir, "case_records")
    os.makedirs(points_vis_dir, exist_ok=True)
    os.makedirs(density_vis_dir, exist_ok=True)
    os.makedirs(case_record_dir, exist_ok=True)

    loaders = {
        "val": get_fsc_loader("val", 1, args),
        "test": get_fsc_loader("test", 1, args),
    }

    model = load_model(args.config, args.pretrain_model, mode="test")
    model = model.to(device)
    model.eval()

    records = []
    for batch in tqdm(loaders[args.load_mode], desc=f"analyze-{args.load_mode}", dynamic_ncols=True):
        records.append(evaluate_one_sample(model, batch, args))

    ranked = sorted(records, key=lambda x: (-x["det_abs_err"], -x["reg_abs_err"], x["file_name"]))

    csv_path = os.path.join(output_dir, "per_image_metrics.csv")
    fieldnames = [
        "rank",
        "file_name",
        "text",
        "gt_count",
        "det_count",
        "reg_count",
        "det_abs_err",
        "reg_abs_err",
        "det_minus_gt",
        "reg_minus_gt",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "split_used",
        "init_query_count",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(ranked, start=1):
            row = {k: record[k] for k in fieldnames if k != "rank"}
            row["rank"] = idx
            writer.writerow(row)

    topk = min(args.topk_vis, len(ranked))
    for idx, record in enumerate(ranked[:topk], start=1):
        stem = f"{idx:03d}_deterr{record['det_abs_err']}_regerr{record['reg_abs_err']:.2f}_{record['file_name']}"
        points_save_path = os.path.join(points_vis_dir, f"{stem}_points.png")
        density_save_path = os.path.join(density_vis_dir, f"{stem}_reg_density.png")
        save_separate_visuals(record, points_save_path, density_save_path)

        h, w = record["image_tensor"].shape[-2:]
        gt_points_abs = (record["gt_points"] * np.array([w, h], dtype=np.float32)).tolist()
        pred_points_abs = (record["pred_points"] * np.array([w, h], dtype=np.float32)).tolist()
        case_record = {
            "rank": idx,
            "file_name": record["file_name"],
            "text": record["text"],
            "gt_count": record["gt_count"],
            "det_count": record["det_count"],
            "reg_count": record["reg_count"],
            "det_abs_err": record["det_abs_err"],
            "reg_abs_err": record["reg_abs_err"],
            "det_minus_gt": record["det_minus_gt"],
            "reg_minus_gt": record["reg_minus_gt"],
            "tp": record["tp"],
            "fp": record["fp"],
            "fn": record["fn"],
            "precision": record["precision"],
            "recall": record["recall"],
            "f1": record["f1"],
            "split_used": record["split_used"],
            "init_query_count": record["init_query_count"],
            "gt_points_xy_abs": gt_points_abs,
            "pred_points_xy_abs": pred_points_abs,
            "point_map_path": points_save_path,
            "reg_density_map_path": density_save_path,
        }
        with open(os.path.join(case_record_dir, f"{stem}.json"), "w") as f:
            json.dump(case_record, f, ensure_ascii=False, indent=2)

    det_mae = sum(r["det_abs_err"] for r in ranked) / max(len(ranked), 1)
    reg_mae = sum(r["reg_abs_err"] for r in ranked) / max(len(ranked), 1)
    split_cases = sum(r["split_used"] for r in ranked)
    over_count_cases = sum(1 for r in ranked if r["det_minus_gt"] > 0)
    under_count_cases = sum(1 for r in ranked if r["det_minus_gt"] < 0)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"split: {args.load_mode}\n")
        f.write(f"eval_impl: {args.eval_impl}\n")
        f.write(f"model: {args.pretrain_model}\n")
        f.write(f"num_images: {len(ranked)}\n")
        f.write(f"det_head_mae: {det_mae:.4f}\n")
        f.write(f"reg_head_mae: {reg_mae:.4f}\n")
        f.write(f"split_used_images: {split_cases}\n")
        f.write(f"over_count_images: {over_count_cases}\n")
        f.write(f"under_count_images: {under_count_cases}\n")
        f.write("top5_worst_det:\n")
        for record in ranked[:5]:
            f.write(
                f"  - {record['file_name']} | gt={record['gt_count']} | det={record['det_count']} | "
                f"reg={record['reg_count']:.2f} | det_err={record['det_abs_err']} | "
                f"reg_err={record['reg_abs_err']:.2f} | split={record['split_used']}\n"
            )

    print(f"Saved per-image metrics to: {csv_path}")
    print(f"Saved top-{topk} point maps to: {points_vis_dir}")
    print(f"Saved top-{topk} reg density maps to: {density_vis_dir}")
    print(f"Saved top-{topk} case records to: {case_record_dir}")
    print(f"Saved summary to: {summary_path}")
