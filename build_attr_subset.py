"""
Build attribute-discriminative subset from REC-8K dataset.

Groups referring expressions by (image_id, class, type) and generates
contrast pairs where the same image has different attribute values for
the same class and attribute type.
"""

import argparse
import json
import os
from collections import defaultdict
from itertools import combinations

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build attribute-discriminative subset and contrast pairs from REC-8K."
    )
    parser.add_argument(
        "--anno_path",
        type=str,
        default="/root/autodl-tmp/data/rec-8k/annotations.json",
        help="Path to annotations.json",
    )
    parser.add_argument(
        "--splits_path",
        type=str,
        default="/root/autodl-tmp/data/rec-8k/splits.json",
        help="Path to splits.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to use (train/val/test)",
    )
    parser.add_argument(
        "--types",
        type=str,
        default="color,action,orientation,age,gender,location",
        help="Comma-separated attribute types to include",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exp/attr_eval/",
        help="Output directory for CSV and JSON files",
    )
    return parser.parse_args()


def get_difficulty_level(gt_diff):
    """Assign difficulty level based on GT count difference."""
    if gt_diff >= 10:
        return 1  # easy
    elif gt_diff >= 3:
        return 2  # medium
    else:
        return 3  # hard


def difficulty_label(level):
    """Human-readable difficulty label."""
    return {1: "easy", 2: "medium", 3: "hard"}.get(level, "unknown")


def main():
    args = parse_args()
    target_types = set(t.strip() for t in args.types.split(","))

    # ---- 1. Load data ----
    with open(args.anno_path, "r") as f:
        annotations = json.load(f)
    with open(args.splits_path, "r") as f:
        splits = json.load(f)

    split_pairs = splits[args.split]  # list of [image_id, caption]
    print(f"Split '{args.split}': {len(split_pairs)} (image_id, caption) pairs")

    # ---- 2. Group by (image_id, class, type) ----
    # Each group entry: {caption, attribute, gt_count}
    groups = defaultdict(list)

    for image_id, caption in split_pairs:
        image_id = str(image_id)
        if image_id not in annotations or caption not in annotations[image_id]:
            continue
        entry = annotations[image_id][caption]
        cls = entry["class"]
        attr = entry["attribute"]
        attr_type = entry["type"]
        points = entry["points"]
        gt_count = len(points)

        if attr_type not in target_types:
            continue

        groups[(image_id, cls, attr_type)].append(
            {
                "caption": caption,
                "attribute": attr,
                "gt_count": gt_count,
            }
        )

    # ---- 3. Filter groups with >= 2 captions and generate pairs ----
    all_entries = []  # flat list of entries in valid groups
    all_pairs = []  # contrast pairs
    # Track max difficulty per (image_id, caption) for entry-level difficulty
    entry_max_difficulty = defaultdict(lambda: 0)  # 0 = no pair yet

    # Per-type statistics
    type_stats = defaultdict(lambda: {"groups": 0, "entries": 0, "pairs": 0})

    for (image_id, cls, attr_type), members in groups.items():
        if len(members) < 2:
            continue

        type_stats[attr_type]["groups"] += 1
        type_stats[attr_type]["entries"] += len(members)

        # Collect entries
        for m in members:
            all_entries.append(
                {
                    "image_id": image_id,
                    "class": cls,
                    "attribute": m["attribute"],
                    "type": attr_type,
                    "caption": m["caption"],
                    "gt_count": m["gt_count"],
                }
            )

        # Generate C(n,2) contrast pairs
        for a, b in combinations(members, 2):
            gt_diff = abs(a["gt_count"] - b["gt_count"])
            level = get_difficulty_level(gt_diff)

            all_pairs.append(
                {
                    "image_id": image_id,
                    "class": cls,
                    "type": attr_type,
                    "attr_a": a["attribute"],
                    "attr_b": b["attribute"],
                    "caption_a": a["caption"],
                    "caption_b": b["caption"],
                    "gt_count_a": a["gt_count"],
                    "gt_count_b": b["gt_count"],
                    "gt_diff": gt_diff,
                    "difficulty_level": level,
                }
            )
            type_stats[attr_type]["pairs"] += 1

            # Update max difficulty for both entries (higher number = harder)
            key_a = (image_id, a["caption"])
            key_b = (image_id, b["caption"])
            entry_max_difficulty[key_a] = max(entry_max_difficulty[key_a], level)
            entry_max_difficulty[key_b] = max(entry_max_difficulty[key_b], level)

    # ---- 4. Assign entry-level difficulty ----
    for e in all_entries:
        key = (e["image_id"], e["caption"])
        e["difficulty_level"] = entry_max_difficulty.get(key, 0)

    # ---- 5. Build DataFrames ----
    df_entries = pd.DataFrame(all_entries)
    df_pairs = pd.DataFrame(all_pairs)

    # Reorder columns
    if not df_entries.empty:
        df_entries = df_entries[
            ["image_id", "class", "attribute", "type", "caption", "gt_count", "difficulty_level"]
        ]
    if not df_pairs.empty:
        df_pairs = df_pairs[
            [
                "image_id", "class", "type", "attr_a", "attr_b",
                "caption_a", "caption_b", "gt_count_a", "gt_count_b",
                "gt_diff", "difficulty_level",
            ]
        ]

    # ---- 6. Difficulty distribution ----
    difficulty_dist = {"easy": 0, "medium": 0, "hard": 0}
    for p in all_pairs:
        difficulty_dist[difficulty_label(p["difficulty_level"])] += 1

    # ---- 7. Build stats summary ----
    stats = {
        "split": args.split,
        "target_types": sorted(target_types),
        "per_type": {},
        "difficulty_distribution": difficulty_dist,
        "total_groups": sum(s["groups"] for s in type_stats.values()),
        "total_entries": len(all_entries),
        "total_pairs": len(all_pairs),
    }
    for t in sorted(type_stats.keys()):
        stats["per_type"][t] = type_stats[t]

    # ---- 8. Save outputs ----
    os.makedirs(args.output_dir, exist_ok=True)

    entries_path = os.path.join(args.output_dir, "attr_subset_entries.csv")
    pairs_path = os.path.join(args.output_dir, "attr_subset_pairs.csv")
    stats_path = os.path.join(args.output_dir, "attr_subset_stats.json")

    df_entries.to_csv(entries_path, index=False)
    df_pairs.to_csv(pairs_path, index=False)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ---- 9. Print summary ----
    print(f"\n{'='*60}")
    print(f"Attribute Subset Summary  (split={args.split})")
    print(f"{'='*60}")
    print(f"{'Type':<15} {'Groups':>8} {'Entries':>8} {'Pairs':>8}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        print(f"{t:<15} {s['groups']:>8} {s['entries']:>8} {s['pairs']:>8}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    print(
        f"{'TOTAL':<15} {stats['total_groups']:>8} "
        f"{stats['total_entries']:>8} {stats['total_pairs']:>8}"
    )
    print(f"\nDifficulty distribution (pairs):")
    print(f"  Level 1 (easy,   gt_diff>=10): {difficulty_dist['easy']}")
    print(f"  Level 2 (medium, 3<=diff<10):  {difficulty_dist['medium']}")
    print(f"  Level 3 (hard,   diff<3):      {difficulty_dist['hard']}")
    print(f"\nOutputs saved to: {os.path.abspath(args.output_dir)}")
    print(f"  - {entries_path}")
    print(f"  - {pairs_path}")
    print(f"  - {stats_path}")


if __name__ == "__main__":
    main()
