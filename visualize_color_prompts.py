import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager as fm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GDDINO_DIR = os.path.join(ROOT_DIR, "GroundingDINO")
if GDDINO_DIR not in sys.path:
    sys.path.insert(0, GDDINO_DIR)

from groundingdino.util.base_api import (
    infer_config_overrides_from_checkpoint,
    load_model,
    load_image,
    threshold,
    preprocess_caption,
)

_PREFERRED_FONTS = [
    "Times New Roman",
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "Times",
]
_AVAILABLE_FONTS = {f.name for f in fm.fontManager.ttflist}
_TEXT_FONT = next((name for name in _PREFERRED_FONTS if name in _AVAILABLE_FONTS), "DejaVu Serif")


def boxes_cxcywh_to_xyxy_abs(boxes, width, height):
    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes_np = boxes.detach().cpu().numpy()
    cx = boxes_np[:, 0] * width
    cy = boxes_np[:, 1] * height
    bw = boxes_np[:, 2] * width
    bh = boxes_np[:, 3] * height
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


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


def boxes_cxcywh_to_centers_abs(boxes, width, height):
    if boxes.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    boxes_np = boxes.detach().cpu().numpy()
    centers = np.stack([boxes_np[:, 0] * width, boxes_np[:, 1] * height], axis=1)
    return centers


def draw_on_axis(
    ax,
    image,
    boxes_xyxy,
    centers_xy,
    logits,
    caption,
    show_scores=True,
    draw_mode="boxes",
    count_label="count",
    gt_count=None,
    title_fontsize=10,
    stat_fontsize=10,
):
    ax.imshow(image)
    if draw_mode == "centers":
        if len(centers_xy) > 0:
            ax.scatter(
                centers_xy[:, 0],
                centers_xy[:, 1],
                c="red",
                marker="*",
                s=36,
                linewidths=0.35,
                edgecolors="black",
            )
    else:
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            rect = patches.Rectangle(
                (x1, y1),
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
                linewidth=1.6,
                edgecolor=(0 / 255, 145 / 255, 210 / 255),
                facecolor="none",
            )
            ax.add_patch(rect)
            if show_scores:
                ax.text(
                    x1,
                    max(0, y1 - 3),
                    f"{float(logits[i]):.2f}",
                    color="white",
                    fontsize=7,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1.2, edgecolor="none"),
                )
    prompt_text = f"\"{caption.rstrip('.')}\""
    if gt_count is None:
        stat_text = f"{count_label}={len(boxes_xyxy)}"
    else:
        stat_text = f"GT={gt_count} | {count_label}={len(boxes_xyxy)}"
    text_font = _TEXT_FONT
    ax.set_title(
        prompt_text,
        fontsize=title_fontsize,
        fontstyle="italic",
        fontfamily=text_font,
        pad=10.0,
    )
    ax.text(
        0.5,
        -0.13,
        stat_text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=stat_fontsize,
        fontfamily=text_font,
        clip_on=False,
    )
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True, help='使用 | 分隔，例如 "red pen|green pen|blue pen"')
    parser.add_argument("--threshold1", type=float, default=0.25)
    parser.add_argument("--threshold2", type=float, default=0.35)
    parser.add_argument("--out_dir", type=str, default="./exp/vis/color_prompts")
    parser.add_argument("--save_each", action="store_true")
    parser.add_argument("--hide_scores", action="store_true")
    parser.add_argument("--draw_mode", type=str, choices=["boxes", "centers"], default="boxes")
    parser.add_argument("--count_label", type=str, default="count")
    parser.add_argument("--gt_counts", type=str, default="", help="使用 | 分隔，与 prompts 对齐，例如 '29|26|1|2'")
    parser.add_argument("--title_fontsize", type=float, default=18.0)
    parser.add_argument("--stat_fontsize", type=float, default=16.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = resolve_overrides(args.checkpoint)
    model = load_model(
        args.config,
        args.checkpoint,
        device=device,
        mode="test",
        config_overrides=overrides,
    ).to(device)
    model.eval()

    image_source, image_tensor = load_image(args.image)
    image_h, image_w = image_source.shape[:2]
    image_batch = image_tensor.unsqueeze(0).to(device)
    prompts = [preprocess_caption(p.strip()) for p in args.prompts.split("|") if p.strip()]
    if len(prompts) == 0:
        raise ValueError("prompts不能为空")
    if args.gt_counts.strip():
        gt_counts = [int(x.strip()) for x in args.gt_counts.split("|") if x.strip()]
        if len(gt_counts) != len(prompts):
            raise ValueError("gt_counts 数量必须与 prompts 数量一致")
    else:
        gt_counts = [None] * len(prompts)

    all_results = []
    for caption, gt_count in zip(prompts, gt_counts):
        with torch.no_grad():
            outputs = model(image_batch, captions=[caption])
        results = threshold(
            outputs,
            [caption],
            model.tokenizer,
            text_threshold=0,
            threshold1=args.threshold1,
            threshold2=args.threshold2,
        )
        boxes, logits, phrases = results[0]
        boxes_xyxy = boxes_cxcywh_to_xyxy_abs(boxes, image_w, image_h)
        centers_xy = boxes_cxcywh_to_centers_abs(boxes, image_w, image_h)
        all_results.append((caption, gt_count, boxes_xyxy, centers_xy, logits.detach().cpu().numpy()))

    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 6), dpi=220)
    if len(all_results) == 1:
        axes = [axes]
    for ax, (caption, gt_count, boxes_xyxy, centers_xy, logits_np) in zip(axes, all_results):
        draw_on_axis(
            ax,
            image_source,
            boxes_xyxy,
            centers_xy,
            logits_np,
            caption,
            show_scores=not args.hide_scores,
            draw_mode=args.draw_mode,
            count_label=args.count_label,
            gt_count=gt_count,
            title_fontsize=args.title_fontsize,
            stat_fontsize=args.stat_fontsize,
        )

    base = os.path.splitext(os.path.basename(args.image))[0]
    compare_path = os.path.join(args.out_dir, f"{base}_color_compare.jpg")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.89, bottom=0.23, wspace=0.08)
    plt.savefig(compare_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    if args.save_each:
        for caption, gt_count, boxes_xyxy, centers_xy, logits_np in all_results:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=220)
            draw_on_axis(
                ax,
                image_source,
                boxes_xyxy,
                centers_xy,
                logits_np,
                caption,
                show_scores=not args.hide_scores,
                draw_mode=args.draw_mode,
                count_label=args.count_label,
                gt_count=gt_count,
                title_fontsize=args.title_fontsize,
                stat_fontsize=args.stat_fontsize,
            )
            name = caption.replace(" ", "_").replace(".", "")
            single_path = os.path.join(args.out_dir, f"{base}_{name}.jpg")
            plt.subplots_adjust(left=0.02, right=0.98, top=0.89, bottom=0.23)
            plt.savefig(single_path, bbox_inches="tight", pad_inches=0.05)
            plt.close()

    print(compare_path)
    for caption, gt_count, boxes_xyxy, _centers_xy, _logits_np in all_results:
        print(f"{caption}: {len(boxes_xyxy)}")


if __name__ == "__main__":
    main()
