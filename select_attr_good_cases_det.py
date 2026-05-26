import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    parser = argparse.ArgumentParser(description="Select top ATTR qualitative good cases using detection-head counts.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py",
    )
    parser.add_argument(
        "--pairs_csv",
        type=str,
        default="./exp/attr_eval/attr_subset_pairs.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exp/analysis/attr_good_cases_det",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/autodl-tmp/data/rec-8k",
    )
    parser.add_argument("--threshold1", type=float, default=0.25)
    parser.add_argument("--threshold2", type=float, default=0.35)
    parser.add_argument("--per_type", type=int, default=4)
    parser.add_argument(
        "--types",
        type=str,
        default="color,gender,action,orientation,age",
        help="comma separated attr types to keep",
    )
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


def boxes_cxcywh_to_xyxy_abs(boxes, width, height):
    if boxes.numel() == 0:
        return []
    boxes_np = boxes.detach().cpu().numpy()
    result = []
    for cx, cy, bw, bh in boxes_np:
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        result.append([x1, y1, x2, y2])
    return result


def run_detection_query(model, image_path, caption, args):
    image_np, image_tensor = load_image(image_path)
    h, w = image_np.shape[:2]
    image_batch = image_tensor.unsqueeze(0).to(args.device if torch.cuda.is_available() else "cpu")
    caption_processed = preprocess_caption(caption)
    with torch.no_grad():
        outputs = model(image_batch, captions=[caption_processed])
    results = threshold(
        outputs,
        [caption_processed],
        model.tokenizer,
        text_threshold=0,
        threshold1=args.threshold1,
        threshold2=args.threshold2,
    )
    boxes, logits, _phrases = results[0]
    boxes_xyxy = boxes_cxcywh_to_xyxy_abs(boxes, w, h)
    return {
        "image_np": image_np,
        "boxes_xyxy": boxes_xyxy,
        "det_count": len(boxes_xyxy),
    }


def draw_pair_visual(image_np, result_a, result_b, row, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=220)
    panels = [
        (
            axes[0],
            row["caption_a"],
            int(row["gt_count_a"]),
            int(result_a["det_count"]),
            result_a["boxes_xyxy"],
        ),
        (
            axes[1],
            row["caption_b"],
            int(row["gt_count_b"]),
            int(result_b["det_count"]),
            result_b["boxes_xyxy"],
        ),
    ]

    for ax, caption, gt_count, det_count, boxes_xyxy in panels:
        ax.imshow(image_np)
        for x1, y1, x2, y2 in boxes_xyxy:
            rect = patches.Rectangle(
                (x1, y1),
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
                linewidth=1.6,
                edgecolor=(0 / 255, 145 / 255, 210 / 255),
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.set_title(f"{caption}\nGT={gt_count} | Det={det_count}", fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"type={row['type']} | diff_gt={int(row['gt_diff'])} | diff_det={int(row['det_diff'])} | "
        f"diff_err={int(row['diff_err'])} | ae_sum={int(row['ae_sum'])}",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    args = parse_args()
    selected_types = [x.strip() for x in args.types.split(",") if x.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    overrides = resolve_overrides(args.checkpoint)
    model = load_model(
        args.config,
        args.checkpoint,
        device=device,
        mode="test",
        config_overrides=overrides,
    )
    model = model.to(device)
    model.eval()

    pairs_df = pd.read_csv(args.pairs_csv)
    pairs_df = pairs_df[pairs_df["type"].isin(selected_types)].copy()
    pairs_df = pairs_df.reset_index(drop=True)

    unique_queries = set()
    for _, row in pairs_df.iterrows():
        unique_queries.add((str(row["image_id"]), row["caption_a"]))
        unique_queries.add((str(row["image_id"]), row["caption_b"]))
    unique_queries = sorted(unique_queries)

    query_cache = {}
    for image_id, caption in tqdm(unique_queries, desc="Detection inference"):
        image_path = os.path.join(args.data_path, image_id)
        query_cache[(image_id, caption)] = run_detection_query(model, image_path, caption, args)

    records = []
    for _, row in pairs_df.iterrows():
        image_id = str(row["image_id"])
        qa = query_cache[(image_id, row["caption_a"])]
        qb = query_cache[(image_id, row["caption_b"])]
        det_a = int(qa["det_count"])
        det_b = int(qb["det_count"])
        gt_a = int(row["gt_count_a"])
        gt_b = int(row["gt_count_b"])
        gt_diff = gt_a - gt_b
        det_diff = det_a - det_b
        order_correct = (gt_diff >= 0 and det_diff >= 0) or (gt_diff < 0 and det_diff < 0)
        diff_err = abs(det_diff - gt_diff)
        ae_sum = abs(det_a - gt_a) + abs(det_b - gt_b)
        records.append(
            {
                **row.to_dict(),
                "det_count_a": det_a,
                "det_count_b": det_b,
                "det_diff": det_diff,
                "diff_err": diff_err,
                "ae_sum": ae_sum,
                "order_correct": int(order_correct),
            }
        )

    ranked_df = pd.DataFrame(records)
    ranked_df = ranked_df[ranked_df["order_correct"] == 1].copy()
    ranked_df = ranked_df.sort_values(
        by=["type", "diff_err", "ae_sum", "difficulty_level", "gt_diff"],
        ascending=[True, True, True, True, False],
    )

    selected_frames = []
    for attr_type in selected_types:
        sub = ranked_df[ranked_df["type"] == attr_type].head(args.per_type).copy()
        sub.insert(0, "rank_in_type", range(1, len(sub) + 1))
        selected_frames.append(sub)

    selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()

    ranked_path = os.path.join(args.output_dir, "ranked_pairs_all.csv")
    selected_path = os.path.join(args.output_dir, "top20_selected.csv")
    ranked_df.to_csv(ranked_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Render selected"):
        attr_type = row["type"]
        image_id = str(row["image_id"])
        out_dir = os.path.join(args.output_dir, attr_type)
        os.makedirs(out_dir, exist_ok=True)
        image_np = query_cache[(image_id, row["caption_a"])]["image_np"]
        result_a = query_cache[(image_id, row["caption_a"])]
        result_b = query_cache[(image_id, row["caption_b"])]
        save_name = (
            f"{int(row['rank_in_type']):02d}_{image_id}"
            f"__{row['attr_a'].replace(' ', '_')}_vs_{row['attr_b'].replace(' ', '_')}.jpg"
        )
        save_path = os.path.join(out_dir, save_name)
        draw_pair_visual(image_np, result_a, result_b, row, save_path)

    print(f"Saved ranked csv to: {ranked_path}")
    print(f"Saved selected csv to: {selected_path}")
    print(f"Saved visuals under: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
