"""
Evaluate attribute-discriminative subset using pre-built CSV files.

Workflow:
  1. Load attr_subset_entries.csv and attr_subset_pairs.csv (from build_attr_subset.py)
  2. Run model inference on each unique (image_id, caption)
  3. Compute AD-MAE / ACR / APDR metrics (from utils/attr_metrics.py)
  4. Output detailed CSV results and JSON report
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup – make GroundingDINO importable
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GDDINO_DIR = os.path.join(ROOT_DIR, "GroundingDINO")
if GDDINO_DIR not in sys.path:
    sys.path.insert(0, GDDINO_DIR)

from groundingdino.util.base_api import (
    infer_config_overrides_from_checkpoint,
    load_model,
    load_image,
    preprocess_caption,
)
from utils.attr_metrics import compute_all_attr_metrics
from utils.trainer_rec import threshold_dynamic_density_query_calibrated


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate attribute-discriminative subset with model inference."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--config", type=str,
                        default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py",
                        help="模型配置文件路径")
    parser.add_argument("--entries_csv", type=str, required=True,
                        help="attr_subset_entries.csv 路径")
    parser.add_argument("--pairs_csv", type=str, required=True,
                        help="attr_subset_pairs.csv 路径")
    parser.add_argument("--output_dir", type=str, default="exp/attr_eval/results/",
                        help="输出目录")
    parser.add_argument("--anno_path", type=str,
                        default="/root/autodl-tmp/data/rec-8k/annotations.json")
    parser.add_argument("--splits_path", type=str,
                        default="/root/autodl-tmp/data/rec-8k/splits.json")
    parser.add_argument("--data_path", type=str,
                        default="/root/autodl-tmp/data/rec-8k/",
                        help="图像数据目录")
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_impl", type=str, default="density", choices=["density", "dagger_calibrated"],
                        help="计数实现：density 为密度积分，dagger_calibrated 为校准版 dagger 选框数")
    parser.add_argument("--dagger_calib_a", type=float, default=1.0,
                        help="calibrated dagger 的缩放系数 a")
    parser.add_argument("--dagger_calib_b", type=float, default=0.0,
                        help="calibrated dagger 的偏置 b")
    parser.add_argument("--threshold1", type=float, default=0.25,
                        help="dagger_calibrated 的 threshold1")
    parser.add_argument("--threshold2", type=float, default=0.35,
                        help="dagger_calibrated 的 threshold2")
    parser.add_argument("--support_k_min", type=int, default=1,
                        help="dagger_calibrated 的最小 top-k")
    parser.add_argument("--support_k_max", type=int, default=900,
                        help="dagger_calibrated 的最大 top-k")
    parser.add_argument("--use_ddg", action="store_true",
                        help="Enable Decoupled Density Guidance")
    parser.add_argument("--use_adm", action="store_true",
                        help="Enable Attribute Decomposition Module")
    parser.add_argument("--use_sdpr", action="store_true",
                        help="Enable SD Prior Density Refinement")
    parser.add_argument("--sd_attn_dir", type=str, default=None,
                        help="预计算的 SD 注意力图目录")
    parser.add_argument("--num_attr_slots", type=int, default=4,
                        help="ADM 属性槽数量")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(model, image_path, caption, scale, device, args):
    """Run single-sample inference and return predicted count (float).

    Args:
        model: loaded GroundingDINO model (eval mode).
        image_path: absolute path to the image file.
        caption: raw caption string.
        scale: density map normalisation factor.
        device: torch device string.

    Returns:
        float: predicted count = sum(density_map) / scale.
    """
    _, image_tensor = load_image(image_path)
    image_batch = image_tensor.unsqueeze(0).to(device)
    caption_processed = preprocess_caption(caption)

    with torch.no_grad():
        outputs = model(image_batch, captions=[caption_processed])

    density = outputs["density"]  # (1, 1, H, W)
    pred_density_count = torch.sum(density).item() / scale
    if args.eval_impl == "dagger_calibrated":
        results = threshold_dynamic_density_query_calibrated(
            outputs,
            [caption_processed],
            model.tokenizer,
            [pred_density_count],
            calib_a=args.dagger_calib_a,
            calib_b=args.dagger_calib_b,
            threshold1=args.threshold1,
            threshold2=args.threshold2,
            k_min=args.support_k_min,
            k_max=args.support_k_max,
        )
        pred_count = float(len(results[0][0])) if len(results) > 0 else 0.0
    else:
        pred_count = pred_density_count

    del outputs, image_batch
    return pred_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- 1. Load CSV files ------------------------------------------------
    entries_df = pd.read_csv(args.entries_csv)
    pairs_df = pd.read_csv(args.pairs_csv)

    print(f"Loaded {len(entries_df)} entries, {len(pairs_df)} pairs")

    # ---- 2. Collect unique (image_id, caption) for inference ---------------
    unique_queries = (
        entries_df[["image_id", "caption"]]
        .drop_duplicates()
        .values.tolist()
    )
    print(f"Unique (image_id, caption) queries: {len(unique_queries)}")

    # ---- 3. Load model ----------------------------------------------------
    print(f"Loading model from {args.checkpoint} ...")
    inferred_overrides = infer_config_overrides_from_checkpoint(args.checkpoint)
    resolved_use_ddg = inferred_overrides.get("use_ddg", False) or args.use_ddg or args.use_sdpr
    resolved_use_sdpr = inferred_overrides.get("use_sdpr", False) or args.use_sdpr
    config_overrides = {
        "use_ddg": resolved_use_ddg,
        "use_adm": args.use_adm,
        "use_sdpr": resolved_use_sdpr,
        "num_attr_slots": args.num_attr_slots,
    }
    print(
        f"Resolved switches -> inferred(ddg={inferred_overrides.get('use_ddg', False)}, "
        f"sdpr={inferred_overrides.get('use_sdpr', False)}), "
        f"final(ddg={resolved_use_ddg}, adm={args.use_adm}, sdpr={resolved_use_sdpr})"
    )
    print(
        f"Attr eval implementation -> {args.eval_impl}"
        + (
            f" (a={args.dagger_calib_a}, b={args.dagger_calib_b}, "
              f"t1={args.threshold1}, t2={args.threshold2})"
            if args.eval_impl == "dagger_calibrated"
            else ""
        )
    )
    model = load_model(
        args.config,
        args.checkpoint,
        device=device,
        mode="test",
        config_overrides=config_overrides,
    )
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # ---- 4. Inference loop ------------------------------------------------
    # Store results: (image_id, caption) -> pred_count
    pred_map = {}
    img_root = args.data_path

    for image_id, caption in tqdm(unique_queries, desc="Inference"):
        image_path = os.path.join(img_root, str(image_id))
        pred_count = run_inference(model, image_path, caption, args.scale, device, args)
        pred_map[(str(image_id), caption)] = pred_count

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 5. Fill predictions into entries_df ------------------------------
    entries_df["pred_count"] = entries_df.apply(
        lambda row: pred_map.get((str(row["image_id"]), row["caption"]), 0.0),
        axis=1,
    )
    entries_df["ae"] = (entries_df["pred_count"] - entries_df["gt_count"]).abs()

    # ---- 6. Fill predictions into pairs_df --------------------------------
    pairs_df["pred_count_a"] = pairs_df.apply(
        lambda row: pred_map.get((str(row["image_id"]), row["caption_a"]), 0.0),
        axis=1,
    )
    pairs_df["pred_count_b"] = pairs_df.apply(
        lambda row: pred_map.get((str(row["image_id"]), row["caption_b"]), 0.0),
        axis=1,
    )
    pairs_df["pred_diff"] = (pairs_df["pred_count_a"] - pairs_df["pred_count_b"]).abs()
    pairs_df["cd_err"] = (
        pairs_df["pred_diff"] - pairs_df["gt_diff"].astype(float)
    ).abs()

    # ---- 7. Compute all metrics -------------------------------------------
    report = compute_all_attr_metrics(entries_df, pairs_df)

    # ---- 8. Save outputs --------------------------------------------------
    entries_out = os.path.join(args.output_dir, "eval_results_entries.csv")
    pairs_out = os.path.join(args.output_dir, "eval_results_pairs.csv")
    report_out = os.path.join(args.output_dir, "eval_report.json")

    entries_df.to_csv(entries_out, index=False)
    pairs_df.to_csv(pairs_out, index=False)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- 9. Print summary -------------------------------------------------
    ov = report["overall"]
    print(f"\n{'='*60}")
    print("Attribute Evaluation Report")
    print(f"{'='*60}")
    print(f"  Entries:  {ov['num_entries']}")
    print(f"  Pairs:    {ov['num_pairs']}")
    print(f"  MAE:      {ov['mae']:.4f}")
    print(f"  RMSE:     {ov['rmse']:.4f}")
    print(f"  AD-MAE:   {ov['ad_mae']:.4f}")
    print(f"  ACR:      {ov['acr']:.4f}")
    print(f"  APDR:     {ov['apdr']:.4f}")

    if report.get("by_type"):
        print(f"\n{'Type':<15} {'MAE':>8} {'AD-MAE':>8} {'ACR':>8} {'APDR':>8}")
        print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for t, m in sorted(report["by_type"].items()):
            print(f"{t:<15} {m['mae']:>8.4f} {m['ad_mae']:>8.4f} "
                  f"{m['acr']:>8.4f} {m['apdr']:>8.4f}")

    if report.get("by_difficulty"):
        print(f"\n{'Difficulty':<15} {'Pairs':>8} {'AD-MAE':>8} {'ACR':>8} {'APDR':>8}")
        print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for d, m in sorted(report["by_difficulty"].items()):
            print(f"{d:<15} {m['num_pairs']:>8} {m['ad_mae']:>8.4f} "
                  f"{m['acr']:>8.4f} {m['apdr']:>8.4f}")

    print(f"\nOutputs saved to: {os.path.abspath(args.output_dir)}")
    print(f"  - {entries_out}")
    print(f"  - {pairs_out}")
    print(f"  - {report_out}")


if __name__ == "__main__":
    main()
