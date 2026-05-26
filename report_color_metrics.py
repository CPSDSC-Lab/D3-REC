import os
import csv
import json
import argparse


def mean(values):
    if len(values) == 0:
        return 0.0
    return float(sum(values) / len(values))


def pda_hits(gt_diff, pred_diff):
    if gt_diff == 0 and pred_diff == 0:
        return 1.0
    if gt_diff == 0 and pred_diff != 0:
        return 0.0
    if gt_diff > 0 and pred_diff > 0:
        return 1.0
    if gt_diff < 0 and pred_diff < 0:
        return 1.0
    return 0.0


def load_rows(csv_path):
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def to_float(rows, key):
    return [float(r[key]) for r in rows]


def build_report(entries_rows, pairs_rows):
    mae_no = mean(to_float(entries_rows, "ae_no_prefilter"))
    mae_yes = mean(to_float(entries_rows, "ae_with_prefilter"))

    cd_no = mean(to_float(pairs_rows, "cd_err_no_prefilter"))
    cd_yes = mean(to_float(pairs_rows, "cd_err_with_prefilter"))

    pda_no = mean([pda_hits(float(r["gt_diff"]), float(r["pred_diff_no_prefilter"])) for r in pairs_rows])
    pda_yes = mean([pda_hits(float(r["gt_diff"]), float(r["pred_diff_with_prefilter"])) for r in pairs_rows])

    bg_pairs = [
        r
        for r in pairs_rows
        if {r["color_a"].lower(), r["color_b"].lower()} == {"blue", "green"}
    ]
    bg_cd_no = mean(to_float(bg_pairs, "cd_err_no_prefilter")) if len(bg_pairs) > 0 else 0.0
    bg_cd_yes = mean(to_float(bg_pairs, "cd_err_with_prefilter")) if len(bg_pairs) > 0 else 0.0
    bg_pda_no = mean([pda_hits(float(r["gt_diff"]), float(r["pred_diff_no_prefilter"])) for r in bg_pairs]) if len(bg_pairs) > 0 else 0.0
    bg_pda_yes = mean([pda_hits(float(r["gt_diff"]), float(r["pred_diff_with_prefilter"])) for r in bg_pairs]) if len(bg_pairs) > 0 else 0.0

    return {
        "entries_count": len(entries_rows),
        "pairs_count": len(pairs_rows),
        "blue_green_pairs_count": len(bg_pairs),
        "metrics": {
            "MAE_color_no_prefilter": mae_no,
            "MAE_color_with_prefilter": mae_yes,
            "CDMAE_no_prefilter": cd_no,
            "CDMAE_with_prefilter": cd_yes,
            "PDA_no_prefilter": pda_no,
            "PDA_with_prefilter": pda_yes,
            "BlueGreen_CDMAE_no_prefilter": bg_cd_no,
            "BlueGreen_CDMAE_with_prefilter": bg_cd_yes,
            "BlueGreen_PDA_no_prefilter": bg_pda_no,
            "BlueGreen_PDA_with_prefilter": bg_pda_yes,
        },
    }


def print_table(report):
    m = report["metrics"]
    lines = [
        ["Metric", "No Prefilter", "With Prefilter", "Delta(With-No)"],
        ["MAE_color", m["MAE_color_no_prefilter"], m["MAE_color_with_prefilter"], m["MAE_color_with_prefilter"] - m["MAE_color_no_prefilter"]],
        ["CD-MAE", m["CDMAE_no_prefilter"], m["CDMAE_with_prefilter"], m["CDMAE_with_prefilter"] - m["CDMAE_no_prefilter"]],
        ["PDA", m["PDA_no_prefilter"], m["PDA_with_prefilter"], m["PDA_with_prefilter"] - m["PDA_no_prefilter"]],
        ["BlueGreen CD-MAE", m["BlueGreen_CDMAE_no_prefilter"], m["BlueGreen_CDMAE_with_prefilter"], m["BlueGreen_CDMAE_with_prefilter"] - m["BlueGreen_CDMAE_no_prefilter"]],
        ["BlueGreen PDA", m["BlueGreen_PDA_no_prefilter"], m["BlueGreen_PDA_with_prefilter"], m["BlueGreen_PDA_with_prefilter"] - m["BlueGreen_PDA_no_prefilter"]],
    ]
    for i, row in enumerate(lines):
        if i == 0:
            print(f"{row[0]:<22} {row[1]:>14} {row[2]:>16} {row[3]:>16}")
        else:
            print(f"{row[0]:<22} {row[1]:>14.4f} {row[2]:>16.4f} {row[3]:>16.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./exp/analysis/color_subset_prefilter_compare")
    parser.add_argument("--entries_csv", type=str, default="color_subset_entries.csv")
    parser.add_argument("--pairs_csv", type=str, default="color_subset_pairs.csv")
    parser.add_argument("--out_json", type=str, default="report_color_metrics.json")
    parser.add_argument("--out_csv", type=str, default="report_color_metrics.csv")
    args = parser.parse_args()

    entries_path = os.path.join(args.input_dir, args.entries_csv)
    pairs_path = os.path.join(args.input_dir, args.pairs_csv)
    out_json_path = os.path.join(args.input_dir, args.out_json)
    out_csv_path = os.path.join(args.input_dir, args.out_csv)

    entries_rows = load_rows(entries_path)
    pairs_rows = load_rows(pairs_path)
    report = build_report(entries_rows, pairs_rows)

    with open(out_json_path, "w") as f:
        json.dump(report, f, indent=2)

    m = report["metrics"]
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "no_prefilter", "with_prefilter", "delta_with_minus_no"])
        writer.writerow(["MAE_color", m["MAE_color_no_prefilter"], m["MAE_color_with_prefilter"], m["MAE_color_with_prefilter"] - m["MAE_color_no_prefilter"]])
        writer.writerow(["CD-MAE", m["CDMAE_no_prefilter"], m["CDMAE_with_prefilter"], m["CDMAE_with_prefilter"] - m["CDMAE_no_prefilter"]])
        writer.writerow(["PDA", m["PDA_no_prefilter"], m["PDA_with_prefilter"], m["PDA_with_prefilter"] - m["PDA_no_prefilter"]])
        writer.writerow(["BlueGreen CD-MAE", m["BlueGreen_CDMAE_no_prefilter"], m["BlueGreen_CDMAE_with_prefilter"], m["BlueGreen_CDMAE_with_prefilter"] - m["BlueGreen_CDMAE_no_prefilter"]])
        writer.writerow(["BlueGreen PDA", m["BlueGreen_PDA_no_prefilter"], m["BlueGreen_PDA_with_prefilter"], m["BlueGreen_PDA_with_prefilter"] - m["BlueGreen_PDA_no_prefilter"]])

    print_table(report)
    print(out_json_path)
    print(out_csv_path)


if __name__ == "__main__":
    main()
