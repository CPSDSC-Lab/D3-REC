import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GDDINO_DIR = os.path.join(ROOT_DIR, "GroundingDINO")
if GDDINO_DIR not in sys.path:
    sys.path.insert(0, GDDINO_DIR)

from groundingdino.util.base_api import (  # noqa: E402
    infer_config_overrides_from_checkpoint,
    load_image,
    load_model,
    preprocess_caption,
    threshold,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch visualize prompt-wise qualitative differences for two checkpoints.")
    parser.add_argument("--checkpoint_a", type=str, required=True)
    parser.add_argument("--checkpoint_b", type=str, required=True)
    parser.add_argument("--label_a", type=str, default="model_a")
    parser.add_argument("--label_b", type=str, default="model_b")
    parser.add_argument("--config", type=str, default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py")
    parser.add_argument("--pairs_csv", type=str, default="./exp/attr_eval/attr_subset_pairs.csv")
    parser.add_argument("--output_dir", type=str, default="./exp/analysis/attr_compare_vis")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/data/rec-8k")
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--threshold1", type=float, default=0.25)
    parser.add_argument("--threshold2", type=float, default=0.35)
    parser.add_argument("--det_mode", type=str, choices=["threshold", "density_topk"], default="density_topk")
    parser.add_argument("--k_min", type=int, default=0)
    parser.add_argument("--k_max", type=int, default=900)
    parser.add_argument("--max_pairs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def resolve_overrides(ckpt_path):
    inferred = infer_config_overrides_from_checkpoint(ckpt_path)
    use_ddg = inferred.get("use_ddg", False) or inferred.get("use_sdpr", False)
    use_sdpr = inferred.get("use_sdpr", False)
    return {
        "use_ddg": use_ddg,
        "use_adm": False,
        "use_sdpr": use_sdpr,
        "num_attr_slots": 4,
    }


def to_abs_points(boxes_xywh, image_w, image_h):
    if boxes_xywh is None or len(boxes_xywh) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = boxes_xywh[:, :2].copy()
    pts[:, 0] *= image_w
    pts[:, 1] *= image_h
    return pts


def run_model_on_queries(args, checkpoint, queries):
    device = args.device if torch.cuda.is_available() else "cpu"
    overrides = resolve_overrides(checkpoint)
    model = load_model(
        args.config,
        checkpoint,
        device=device,
        mode="test",
        config_overrides=overrides,
    )
    model = model.to(device)
    model.eval()

    pred_map = {}
    for image_id, caption in tqdm(queries, desc=f"Infer {os.path.basename(checkpoint)}"):
        image_path = os.path.join(args.data_path, str(image_id))
        image_np, image_tensor = load_image(image_path)
        h, w = image_np.shape[:2]

        cap = preprocess_caption(caption)
        with torch.no_grad():
            outputs = model(image_tensor.unsqueeze(0).to(device), captions=[cap])
        pred_count = float(torch.sum(outputs["density"]).item() / args.scale)

        det_res = threshold(
            outputs,
            [cap],
            model.tokenizer,
            text_threshold=0.0,
            threshold1=args.threshold1,
            threshold2=args.threshold2,
        )
        if len(det_res) > 0:
            boxes_t = det_res[0][0]
            scores_t = det_res[0][1]
        else:
            boxes_t = torch.zeros((0, 4))
            scores_t = torch.zeros((0,))

        if args.det_mode == "density_topk" and boxes_t.shape[0] > 0:
            topk = int(round(pred_count))
            topk = max(int(args.k_min), min(int(args.k_max), topk))
            topk = min(topk, int(boxes_t.shape[0]))
            if topk > 0:
                scores_t = scores_t.reshape(-1)
                _, keep_idx = torch.topk(scores_t, topk, dim=0)
                keep_idx = keep_idx.reshape(-1)
                boxes_t = boxes_t.index_select(0, keep_idx)
                scores_t = scores_t.index_select(0, keep_idx)
            else:
                boxes_t = boxes_t[:0]
                scores_t = scores_t[:0]

        boxes = boxes_t.cpu().numpy() if boxes_t.numel() > 0 else np.zeros((0, 4), dtype=np.float32)
        points_abs = to_abs_points(boxes, w, h)

        pred_map[(str(image_id), caption)] = {
            "pred_count": pred_count,
            "num_queries": int(points_abs.shape[0]),
            "points_abs": points_abs.tolist(),
        }

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred_map


def draw_panel(ax, image_np, points_abs, title):
    ax.imshow(image_np)
    if len(points_abs) > 0:
        pts = np.asarray(points_abs)
        ax.scatter(pts[:, 0], pts[:, 1], c="deepskyblue", marker="*", s=35, linewidths=0.3, edgecolors="black")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    pairs_df = pd.read_csv(args.pairs_csv)
    if args.max_pairs > 0 and len(pairs_df) > args.max_pairs:
        pairs_df = pairs_df.sample(n=args.max_pairs, random_state=args.seed).reset_index(drop=True)

    unique_queries = set()
    for _, row in pairs_df.iterrows():
        unique_queries.add((str(row["image_id"]), row["caption_a"]))
        unique_queries.add((str(row["image_id"]), row["caption_b"]))
    unique_queries = sorted(list(unique_queries))
    print(f"Pairs for visualization: {len(pairs_df)}")
    print(f"Unique queries: {len(unique_queries)}")

    pred_a = run_model_on_queries(args, args.checkpoint_a, unique_queries)
    pred_b = run_model_on_queries(args, args.checkpoint_b, unique_queries)

    summary_rows = []
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Render"):
        image_id = str(row["image_id"])
        caption_a = row["caption_a"]
        caption_b = row["caption_b"]
        gt_a = float(row["gt_count_a"])
        gt_b = float(row["gt_count_b"])
        attr_type = row.get("type", "unknown")

        image_path = os.path.join(args.data_path, image_id)
        image_np, _ = load_image(image_path)

        pa_a = pred_a[(image_id, caption_a)]
        pb_a = pred_b[(image_id, caption_a)]
        pa_b = pred_a[(image_id, caption_b)]
        pb_b = pred_b[(image_id, caption_b)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
        fig.suptitle(
            f"[{idx:04d}] image={image_id} | type={attr_type}\n"
            f"A: \"{caption_a}\" (gt={gt_a:.1f}) | B: \"{caption_b}\" (gt={gt_b:.1f})",
            fontsize=11,
        )

        draw_panel(
            axes[0, 0],
            image_np,
            pa_a["points_abs"],
            f"{args.label_a} | pred={pa_a['pred_count']:.2f} | det_q={pa_a['num_queries']}",
        )
        draw_panel(
            axes[0, 1],
            image_np,
            pb_a["points_abs"],
            f"{args.label_b} | pred={pb_a['pred_count']:.2f} | det_q={pb_a['num_queries']}",
        )
        draw_panel(
            axes[1, 0],
            image_np,
            pa_b["points_abs"],
            f"{args.label_a} | pred={pa_b['pred_count']:.2f} | det_q={pa_b['num_queries']}",
        )
        draw_panel(
            axes[1, 1],
            image_np,
            pb_b["points_abs"],
            f"{args.label_b} | pred={pb_b['pred_count']:.2f} | det_q={pb_b['num_queries']}",
        )

        out_img = os.path.join(args.output_dir, f"{idx:04d}_{image_id}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(out_img)
        plt.close(fig)

        summary_rows.append(
            {
                "index": idx,
                "image_id": image_id,
                "type": attr_type,
                "caption_a": caption_a,
                "caption_b": caption_b,
                "gt_count_a": gt_a,
                "gt_count_b": gt_b,
                f"pred_a_{args.label_a}": pa_a["pred_count"],
                f"pred_a_{args.label_b}": pb_a["pred_count"],
                f"pred_b_{args.label_a}": pa_b["pred_count"],
                f"pred_b_{args.label_b}": pb_b["pred_count"],
            }
        )

    summary_path = os.path.join(args.output_dir, "compare_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved visualizations to: {os.path.abspath(args.output_dir)}")
    print(f"Saved summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
