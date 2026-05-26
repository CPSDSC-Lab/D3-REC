import os
import sys
import re
import csv
import json
import itertools
import argparse
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GDDINO_DIR = os.path.join(ROOT_DIR, "GroundingDINO")
if GDDINO_DIR not in sys.path:
    sys.path.insert(0, GDDINO_DIR)

from groundingdino.util.base_api import load_model, load_image
from groundingdino.util.base_api import preprocess_caption
from utils.processor import DataProcessor
from utils.tester_rec import threshold_dynamic_context


def find_color(caption, colors):
    low = caption.lower()
    for c in colors:
        if re.search(rf"\b{re.escape(c)}\b", low):
            return c
    return None


def mean_or_zero(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--colors", type=str, default="red,green,blue,yellow,black,white")
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--prefilter_ratio", type=float, default=0.08)
    parser.add_argument("--out_dir", type=str, default="./exp/color_subset_eval")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    colors = [x.strip().lower() for x in args.colors.split(",") if x.strip()]

    processor = DataProcessor()
    split_pairs = processor.splits[args.split]
    ann = processor.annotations

    groups = {}
    for image_id, caption in split_pairs:
        item = ann[(image_id, caption)]
        cls = str(item.get("class", "")).strip().lower()
        if cls == "":
            continue
        color = find_color(caption, colors)
        if color is None:
            continue
        key = (image_id, cls)
        if key not in groups:
            groups[key] = {}
        if color not in groups[key]:
            points = item.get("points", [])
            gt_count = int(len(points))
            groups[key][color] = {"caption": caption, "gt_count": gt_count}

    subset = []
    for (image_id, cls), cmap in groups.items():
        if len(cmap) < 2:
            continue
        for color, payload in cmap.items():
            subset.append(
                {
                    "image_id": image_id,
                    "class": cls,
                    "color": color,
                    "caption": payload["caption"],
                    "gt_count": payload["gt_count"],
                }
            )

    by_image = {}
    for row in subset:
        by_image.setdefault((row["image_id"], row["class"]), []).append(row)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.config, args.checkpoint, mode="test").to(device)
    model.eval()

    entry_rows = []
    pair_rows = []
    img_root = processor.get_image_path()

    for (image_id, cls), entries in by_image.items():
        image_path = os.path.join(img_root, image_id)
        image_source, image_tensor = load_image(image_path)
        image_batch = torch.stack([image_tensor for _ in range(len(entries))], dim=0).to(device)
        captions = [preprocess_caption(e["caption"]) for e in entries]
        with torch.no_grad():
            outputs = model(image_batch, captions=captions)
        pred_num = torch.sum(outputs["density"].squeeze(1), dim=[1, 2]) / args.scale
        pred_num_round = [round(x.item()) for x in pred_num]

        results_no = threshold_dynamic_context(
            outputs,
            captions,
            image_batch,
            model.tokenizer,
            pred_num_round,
            color_prefilter=False,
        )
        results_yes = threshold_dynamic_context(
            outputs,
            captions,
            image_batch,
            model.tokenizer,
            pred_num_round,
            color_prefilter=True,
            color_prefilter_min_ratio=args.prefilter_ratio,
        )

        for i, e in enumerate(entries):
            pred_no = int(results_no[i][0].shape[0])
            pred_yes = int(results_yes[i][0].shape[0])
            gt = int(e["gt_count"])
            row = {
                "image_id": image_id,
                "class": cls,
                "color": e["color"],
                "caption": e["caption"],
                "gt_count": gt,
                "pred_no_prefilter": pred_no,
                "pred_with_prefilter": pred_yes,
                "ae_no_prefilter": abs(pred_no - gt),
                "ae_with_prefilter": abs(pred_yes - gt),
            }
            entry_rows.append(row)

        for i, j in itertools.combinations(range(len(entries)), 2):
            a = entries[i]
            b = entries[j]
            gt_diff = int(a["gt_count"]) - int(b["gt_count"])
            no_diff = int(results_no[i][0].shape[0]) - int(results_no[j][0].shape[0])
            yes_diff = int(results_yes[i][0].shape[0]) - int(results_yes[j][0].shape[0])
            pair_rows.append(
                {
                    "image_id": image_id,
                    "class": cls,
                    "color_a": a["color"],
                    "color_b": b["color"],
                    "gt_diff": gt_diff,
                    "pred_diff_no_prefilter": no_diff,
                    "pred_diff_with_prefilter": yes_diff,
                    "cd_err_no_prefilter": abs(no_diff - gt_diff),
                    "cd_err_with_prefilter": abs(yes_diff - gt_diff),
                }
            )

        del outputs, results_no, results_yes, image_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mae_no = mean_or_zero([x["ae_no_prefilter"] for x in entry_rows])
    mae_yes = mean_or_zero([x["ae_with_prefilter"] for x in entry_rows])
    cd_no = mean_or_zero([x["cd_err_no_prefilter"] for x in pair_rows])
    cd_yes = mean_or_zero([x["cd_err_with_prefilter"] for x in pair_rows])

    summary = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "colors": colors,
        "subset_groups": len(by_image),
        "subset_entries": len(entry_rows),
        "subset_pairs": len(pair_rows),
        "mae_no_prefilter": mae_no,
        "mae_with_prefilter": mae_yes,
        "cdmae_no_prefilter": cd_no,
        "cdmae_with_prefilter": cd_yes,
        "prefilter_ratio": args.prefilter_ratio,
    }

    entries_csv = os.path.join(args.out_dir, "color_subset_entries.csv")
    with open(entries_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(entry_rows[0].keys()) if entry_rows else [
            "image_id", "class", "color", "caption", "gt_count", "pred_no_prefilter", "pred_with_prefilter", "ae_no_prefilter", "ae_with_prefilter"
        ])
        writer.writeheader()
        if entry_rows:
            writer.writerows(entry_rows)

    pairs_csv = os.path.join(args.out_dir, "color_subset_pairs.csv")
    with open(pairs_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()) if pair_rows else [
            "image_id", "class", "color_a", "color_b", "gt_diff", "pred_diff_no_prefilter", "pred_diff_with_prefilter", "cd_err_no_prefilter", "cd_err_with_prefilter"
        ])
        writer.writeheader()
        if pair_rows:
            writer.writerows(pair_rows)

    summary_json = os.path.join(args.out_dir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(entries_csv)
    print(pairs_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
