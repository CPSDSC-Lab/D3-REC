"""
属性区分评估报告生成脚本。

支持两种模式：
  - 单模型报告：--results_dir 指定一个结果目录
  - 多模型对比：--results_dirs 指定多个结果目录（逗号分隔）
"""

import os
import json
import argparse

import pandas as pd


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_report(results_dir):
    """优先从 eval_report.json 读取，否则从 CSV 重新计算。"""
    json_path = os.path.join(results_dir, "eval_report.json")
    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    # fallback: 从 CSV 计算
    from utils.attr_metrics import compute_all_attr_metrics

    entries_path = os.path.join(results_dir, "eval_results_entries.csv")
    pairs_path = os.path.join(results_dir, "eval_results_pairs.csv")
    entries_df = pd.read_csv(entries_path)
    pairs_df = pd.read_csv(pairs_path)
    return compute_all_attr_metrics(entries_df, pairs_df)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_f2(v):
    """数值保留 2 位小数，无数据时显示 '-'。"""
    if v is None:
        return "-"
    return "{:.2f}".format(v)


def _fmt_pct1(v):
    """百分比保留 1 位小数，无数据时显示 '-'。"""
    if v is None:
        return "-"
    return "{:.1f}".format(v * 100)


def _fmt_int(v):
    if v is None:
        return "-"
    return str(int(v))


# ---------------------------------------------------------------------------
# 单模型报告
# ---------------------------------------------------------------------------

ATTR_TYPES_ORDER = ["color", "action", "orientation", "age", "gender", "location"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def print_single_report(report):
    """终端打印格式化的单模型报告。"""
    o = report["overall"]
    by_type = report.get("by_type", {})
    by_diff = report.get("by_difficulty", {})

    print()
    print("================== 属性区分评估报告 ==================")
    print()
    print("总体指标:")
    print("  条目数: {}  | 对比对数: {}".format(
        _fmt_int(o.get("num_entries")), _fmt_int(o.get("num_pairs"))))
    print("  MAE: {}  | RMSE: {}".format(
        _fmt_f2(o.get("mae")), _fmt_f2(o.get("rmse"))))
    print("  AD-MAE: {}  | ACR: {}%  | APDR: {}%".format(
        _fmt_f2(o.get("ad_mae")),
        _fmt_pct1(o.get("acr")),
        _fmt_pct1(o.get("apdr"))))

    # 按属性类型
    print()
    print("按属性类型:")
    header = "  {:<14s} {:>6s} {:>6s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "类型", "条目数", "对数", "MAE", "AD-MAE", "ACR(%)", "APDR(%)")
    print(header)

    # 收集所有出现的类型（保持指定顺序，额外类型追加到末尾）
    all_types = list(ATTR_TYPES_ORDER)
    for t in sorted(by_type.keys()):
        if t not in all_types:
            all_types.append(t)

    for t in all_types:
        if t not in by_type:
            continue
        d = by_type[t]
        # num_pairs 可能在 by_type 中不存在（compute_all_attr_metrics 未存），从 by_type_difficulty 推算
        num_pairs = d.get("num_pairs")
        if num_pairs is None:
            # 尝试从 by_type_difficulty 求和
            td = report.get("by_type_difficulty", {}).get(t, {})
            if td:
                num_pairs = sum(v.get("num_pairs", 0) for v in td.values())

        print("  {:<14s} {:>6s} {:>6s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
            t,
            _fmt_int(d.get("num_entries")),
            _fmt_int(num_pairs),
            _fmt_f2(d.get("mae")),
            _fmt_f2(d.get("ad_mae")),
            _fmt_pct1(d.get("acr")),
            _fmt_pct1(d.get("apdr")),
        ))

    # 按难度分级
    print()
    print("按难度分级:")
    print("  {:<10s} {:>6s} {:>8s} {:>8s} {:>8s}".format(
        "难度", "对数", "AD-MAE", "ACR(%)", "APDR(%)"))

    for lvl in DIFFICULTY_ORDER:
        if lvl not in by_diff:
            continue
        d = by_diff[lvl]
        print("  {:<10s} {:>6s} {:>8s} {:>8s} {:>8s}".format(
            lvl,
            _fmt_int(d.get("num_pairs")),
            _fmt_f2(d.get("ad_mae")),
            _fmt_pct1(d.get("acr")),
            _fmt_pct1(d.get("apdr")),
        ))
    # 追加非标准难度
    for lvl in sorted(by_diff.keys()):
        if lvl not in DIFFICULTY_ORDER:
            d = by_diff[lvl]
            print("  {:<10s} {:>6s} {:>8s} {:>8s} {:>8s}".format(
                lvl,
                _fmt_int(d.get("num_pairs")),
                _fmt_f2(d.get("ad_mae")),
                _fmt_pct1(d.get("acr")),
                _fmt_pct1(d.get("apdr")),
            ))
    print()


def save_single_report(report, output_dir):
    """保存单模型报告的 JSON 和 CSV 文件。"""
    os.makedirs(output_dir, exist_ok=True)

    # report_summary.json
    summary_path = os.path.join(output_dir, "report_summary.json")
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("已保存: {}".format(summary_path))

    # report_by_type.csv
    by_type = report.get("by_type", {})
    by_td = report.get("by_type_difficulty", {})
    rows_type = []
    all_types = list(ATTR_TYPES_ORDER)
    for t in sorted(by_type.keys()):
        if t not in all_types:
            all_types.append(t)
    for t in all_types:
        if t not in by_type:
            continue
        d = by_type[t]
        num_pairs = d.get("num_pairs")
        if num_pairs is None:
            td = by_td.get(t, {})
            num_pairs = sum(v.get("num_pairs", 0) for v in td.values()) if td else 0
        rows_type.append({
            "type": t,
            "num_entries": d.get("num_entries", 0),
            "num_pairs": num_pairs if num_pairs is not None else 0,
            "mae": round(d.get("mae", 0.0), 2),
            "ad_mae": round(d.get("ad_mae", 0.0), 2),
            "acr": round(d.get("acr", 0.0) * 100, 1),
            "apdr": round(d.get("apdr", 0.0) * 100, 1),
        })
    type_csv_path = os.path.join(output_dir, "report_by_type.csv")
    pd.DataFrame(rows_type).to_csv(type_csv_path, index=False)
    print("已保存: {}".format(type_csv_path))

    # report_by_difficulty.csv
    by_diff = report.get("by_difficulty", {})
    rows_diff = []
    for lvl in DIFFICULTY_ORDER:
        if lvl not in by_diff:
            continue
        d = by_diff[lvl]
        rows_diff.append({
            "difficulty": lvl,
            "num_pairs": d.get("num_pairs", 0),
            "ad_mae": round(d.get("ad_mae", 0.0), 2),
            "acr": round(d.get("acr", 0.0) * 100, 1),
            "apdr": round(d.get("apdr", 0.0) * 100, 1),
        })
    for lvl in sorted(by_diff.keys()):
        if lvl not in DIFFICULTY_ORDER:
            d = by_diff[lvl]
            rows_diff.append({
                "difficulty": lvl,
                "num_pairs": d.get("num_pairs", 0),
                "ad_mae": round(d.get("ad_mae", 0.0), 2),
                "acr": round(d.get("acr", 0.0) * 100, 1),
                "apdr": round(d.get("apdr", 0.0) * 100, 1),
            })
    diff_csv_path = os.path.join(output_dir, "report_by_difficulty.csv")
    pd.DataFrame(rows_diff).to_csv(diff_csv_path, index=False)
    print("已保存: {}".format(diff_csv_path))


# ---------------------------------------------------------------------------
# 多模型对比
# ---------------------------------------------------------------------------

def print_comparison(reports, model_names):
    """终端打印多模型对比表。"""
    n = len(model_names)
    col_w = max(12, max(len(name) for name in model_names) + 2)

    print()
    print("=================== 多模型对比 ===================")

    # 总体对比
    print()
    print("总体对比:")
    header = "  {:<10s}".format("指标")
    for name in model_names:
        header += " {:>{w}s}".format(name, w=col_w)
    print(header)

    metrics_info = [
        ("MAE", lambda r: r["overall"].get("mae"), _fmt_f2),
        ("RMSE", lambda r: r["overall"].get("rmse"), _fmt_f2),
        ("AD-MAE", lambda r: r["overall"].get("ad_mae"), _fmt_f2),
        ("ACR(%)", lambda r: r["overall"].get("acr"), _fmt_pct1),
        ("APDR(%)", lambda r: r["overall"].get("apdr"), _fmt_pct1),
    ]

    for label, getter, fmt_fn in metrics_info:
        row = "  {:<10s}".format(label)
        for rpt in reports:
            row += " {:>{w}s}".format(fmt_fn(getter(rpt)), w=col_w)
        print(row)

    # 按类型 AD-MAE 对比
    print()
    print("按类型 AD-MAE 对比:")
    header = "  {:<14s}".format("类型")
    for name in model_names:
        header += " {:>{w}s}".format(name, w=col_w)
    print(header)

    all_types = set()
    for rpt in reports:
        all_types.update(rpt.get("by_type", {}).keys())
    ordered_types = [t for t in ATTR_TYPES_ORDER if t in all_types]
    for t in sorted(all_types):
        if t not in ordered_types:
            ordered_types.append(t)

    for t in ordered_types:
        row = "  {:<14s}".format(t)
        for rpt in reports:
            v = rpt.get("by_type", {}).get(t, {}).get("ad_mae")
            row += " {:>{w}s}".format(_fmt_f2(v), w=col_w)
        print(row)

    # 按类型 ACR 对比
    print()
    print("按类型 ACR(%) 对比:")
    header = "  {:<14s}".format("类型")
    for name in model_names:
        header += " {:>{w}s}".format(name, w=col_w)
    print(header)
    for t in ordered_types:
        row = "  {:<14s}".format(t)
        for rpt in reports:
            v = rpt.get("by_type", {}).get(t, {}).get("acr")
            row += " {:>{w}s}".format(_fmt_pct1(v), w=col_w)
        print(row)

    # 按类型 APDR 对比
    print()
    print("按类型 APDR(%) 对比:")
    header = "  {:<14s}".format("类型")
    for name in model_names:
        header += " {:>{w}s}".format(name, w=col_w)
    print(header)
    for t in ordered_types:
        row = "  {:<14s}".format(t)
        for rpt in reports:
            v = rpt.get("by_type", {}).get(t, {}).get("apdr")
            row += " {:>{w}s}".format(_fmt_pct1(v), w=col_w)
        print(row)

    print()


def save_comparison(reports, model_names, output_dir):
    """保存多模型对比 CSV。"""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    overall_metrics = ["mae", "rmse", "ad_mae", "acr", "apdr"]
    for m in overall_metrics:
        row = {"metric": m, "scope": "overall"}
        for name, rpt in zip(model_names, reports):
            v = rpt["overall"].get(m, 0.0)
            if m in ("acr", "apdr"):
                v = round(v * 100, 1)
            else:
                v = round(v, 2)
            row[name] = v
        rows.append(row)

    # by_type breakdown
    all_types = set()
    for rpt in reports:
        all_types.update(rpt.get("by_type", {}).keys())
    ordered_types = [t for t in ATTR_TYPES_ORDER if t in all_types]
    for t in sorted(all_types):
        if t not in ordered_types:
            ordered_types.append(t)

    for t in ordered_types:
        for m in ["mae", "ad_mae", "acr", "apdr"]:
            row = {"metric": m, "scope": "type/" + t}
            for name, rpt in zip(model_names, reports):
                v = rpt.get("by_type", {}).get(t, {}).get(m)
                if v is None:
                    row[name] = ""
                elif m in ("acr", "apdr"):
                    row[name] = round(v * 100, 1)
                else:
                    row[name] = round(v, 2)
            rows.append(row)

    csv_path = os.path.join(output_dir, "report_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("已保存: {}".format(csv_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="属性区分评估报告生成")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="单个结果目录路径")
    parser.add_argument("--results_dirs", type=str, default=None,
                        help="多个结果目录，逗号分隔")
    parser.add_argument("--model_names", type=str, default=None,
                        help="模型名称，逗号分隔")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="报告输出目录（默认与 results_dir 相同）")
    args = parser.parse_args()

    # 参数校验
    if args.results_dirs is not None:
        # 多模型对比模式
        dirs = [d.strip() for d in args.results_dirs.split(",") if d.strip()]
        if len(dirs) < 2:
            parser.error("--results_dirs 至少需要提供 2 个目录")

        if args.model_names is not None:
            names = [n.strip() for n in args.model_names.split(",") if n.strip()]
        else:
            names = [os.path.basename(os.path.normpath(d)) for d in dirs]

        if len(names) != len(dirs):
            parser.error("--model_names 数量({})与 --results_dirs 数量({})不匹配".format(
                len(names), len(dirs)))

        output_dir = args.output_dir if args.output_dir else dirs[0]

        reports = []
        for d in dirs:
            print("加载结果: {}".format(d))
            reports.append(load_report(d))

        print_comparison(reports, names)
        save_comparison(reports, names, output_dir)

    elif args.results_dir is not None:
        # 单模型报告模式
        output_dir = args.output_dir if args.output_dir else args.results_dir

        print("加载结果: {}".format(args.results_dir))
        report = load_report(args.results_dir)

        print_single_report(report)
        save_single_report(report, output_dir)

    else:
        parser.error("请提供 --results_dir（单模型）或 --results_dirs（多模型对比）")


if __name__ == "__main__":
    main()
