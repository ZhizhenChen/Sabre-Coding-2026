"""
Test script for lambda_model distribution outputs.
Loads 3,000 requests from 2026-02-07 and writes lambda/TTL distributions
for each lambda estimation method into a txt report.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_processing.pipeline import DataPipelineProcessor
from Lambda.lambda_model import build_lambda_table, ttl_from_lambda


METHODS = ["km", "poisson", "glm", "fallback"]
TARGET_FRESHNESS = 0.8


def _format_series(series: pd.Series, float_digits: int = 8) -> str:
    lines: list[str] = []
    for idx, val in series.items():
        if isinstance(val, (float, np.floating, int, np.integer)):
            lines.append(f"  {idx}: {float(val):.{float_digits}f}")
        else:
            lines.append(f"  {idx}: {val}")
    return "\n".join(lines)


def _distribution_bins(series: pd.Series, bins: list[float], labels: list[str]) -> tuple[pd.Series, pd.Series]:
    clipped = series.clip(lower=bins[0])
    bucketed = pd.cut(clipped, bins=bins, labels=labels, include_lowest=True, right=True)
    counts = bucketed.value_counts().reindex(labels, fill_value=0)
    pct = (counts / max(len(series), 1)) * 100.0
    return counts, pct


def _build_method_block(method: str, table: pd.DataFrame, split_label: str = "all") -> str:
    lines: list[str] = []

    lambda_series = pd.to_numeric(table.get("lambda_final"), errors="coerce")
    lambda_series = lambda_series[np.isfinite(lambda_series) & (lambda_series > 0)]

    ttl_series = lambda_series.apply(lambda x: ttl_from_lambda(float(x)))

    lines.append("=" * 100)
    lines.append(f"METHOD: {method} (split={split_label})")
    lines.append("=" * 100)
    lines.append(f"rows_in_lambda_table: {len(table)}")
    lines.append(f"valid_lambda_count: {len(lambda_series)}")
    lines.append(f"valid_ttl_count: {len(ttl_series)}")

    if len(lambda_series) == 0:
        lines.append("No valid lambda values for this method.")
        lines.append("")
        return "\n".join(lines)

    lambda_desc = lambda_series.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    ttl_desc = ttl_series.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    lines.append("")
    lines.append("LAMBDA SUMMARY")
    lines.append("-" * 100)
    lines.append(_format_series(lambda_desc, float_digits=10))

    lines.append("")
    lines.append("TTL (SECONDS) SUMMARY")
    lines.append("-" * 100)
    lines.append(_format_series(ttl_desc, float_digits=4))

    lambda_bins = [0.0, 0.001, 0.01, 0.05, 0.10, 0.50, 1.0, 5.0, 10.0, float("inf")]
    lambda_labels = [
        "[0,0.001]",
        "(0.001,0.01]",
        "(0.01,0.05]",
        "(0.05,0.10]",
        "(0.10,0.50]",
        "(0.50,1.00]",
        "(1.00,5.00]",
        "(5.00,10.00]",
        ">(10.00]",
    ]
    lambda_counts, lambda_pct = _distribution_bins(lambda_series, lambda_bins, lambda_labels)

    ttl_bins = [60, 300, 600, 1800, 3600, 7200, 21600, 43200, 86400]
    ttl_labels = [
        "[60,300]",
        "(300,600]",
        "(600,1800]",
        "(1800,3600]",
        "(3600,7200]",
        "(7200,21600]",
        "(21600,43200]",
        "(43200,86400]",
    ]
    ttl_counts, ttl_pct = _distribution_bins(ttl_series.astype(float), ttl_bins, ttl_labels)

    lines.append("")
    lines.append("LAMBDA BIN DISTRIBUTION")
    lines.append("-" * 100)
    for label in lambda_labels:
        lines.append(f"  {label:14s}: {int(lambda_counts[label]):6d} ({float(lambda_pct[label]):6.2f}%)")

    lines.append("")
    lines.append("TTL BIN DISTRIBUTION (seconds)")
    lines.append("-" * 100)
    for label in ttl_labels:
        lines.append(f"  {label:14s}: {int(ttl_counts[label]):6d} ({float(ttl_pct[label]):6.2f}%)")

    lines.append("")
    return "\n".join(lines)


def _build_lambda_diagnostic_block(method: str, table: pd.DataFrame, split_label: str) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"LAMBDA DIAGNOSTIC: {method} (split={split_label})")
    lines.append("=" * 100)

    lam = pd.to_numeric(table.get("lambda_final"), errors="coerce")
    lam = lam[np.isfinite(lam) & (lam > 0)]
    n = int(len(lam))
    lines.append(f"n_valid_lambda: {n}")
    if n == 0:
        lines.append("No valid lambda values for diagnostics.")
        lines.append("")
        return "\n".join(lines)

    p99 = float(lam.quantile(0.99))
    p999 = float(lam.quantile(0.999))
    p95 = float(lam.quantile(0.95))
    std_raw = float(lam.std(ddof=1)) if n > 1 else 0.0
    std_clip_p99 = float(lam.clip(upper=p99).std(ddof=1)) if n > 1 else 0.0
    std_clip_p999 = float(lam.clip(upper=p999).std(ddof=1)) if n > 1 else 0.0
    mean_raw = float(lam.mean())
    median_raw = float(lam.median())

    lines.append(f"lambda_mean: {mean_raw:.8f}")
    lines.append(f"lambda_median: {median_raw:.8f}")
    lines.append(f"lambda_std_raw: {std_raw:.8f}")
    lines.append(f"lambda_std_clip_p99: {std_clip_p99:.8f}")
    lines.append(f"lambda_std_clip_p999: {std_clip_p999:.8f}")
    lines.append(f"lambda_p95: {p95:.8f}")
    lines.append(f"lambda_p99: {p99:.8f}")
    lines.append(f"lambda_p999: {p999:.8f}")
    lines.append(f"share_lambda_gt_1: {float((lam > 1).mean()):.6f}")
    lines.append(f"share_lambda_gt_10: {float((lam > 10).mean()):.6f}")
    lines.append(f"share_lambda_gt_100: {float((lam > 100).mean()):.6f}")

    if "_in_lookup" in table.columns:
        in_lookup = table["_in_lookup"].astype(bool)
        lines.append(f"lookup_eligible_ratio(_in_lookup=True): {float(in_lookup.mean()):.6f}")

    if "_global_lambda" in table.columns:
        global_col = pd.to_numeric(table["_global_lambda"], errors="coerce")
        pair = pd.DataFrame({"lam": pd.to_numeric(table.get("lambda_final"), errors="coerce"), "glob": global_col})
        pair = pair[np.isfinite(pair["lam"]) & np.isfinite(pair["glob"])]
        if not pair.empty:
            # Treat values extremely close to global as global-fill rows.
            near_global = np.isclose(pair["lam"], pair["glob"], rtol=1e-10, atol=1e-12)
            lines.append(f"global_fill_ratio(lambda_final==global_lambda): {float(np.mean(near_global)):.6f}")

    if method == "glm" and "lambda_glm" in table.columns:
        glm_lam = pd.to_numeric(table["lambda_glm"], errors="coerce")
        glm_valid = glm_lam[np.isfinite(glm_lam) & (glm_lam > 0)]
        lines.append(f"glm_direct_valid_ratio(lambda_glm>0): {float(len(glm_valid) / max(len(table), 1)):.6f}")
        if "lambda_method" in table.columns:
            fallback_ratio = float(table["lambda_method"].astype(str).str.contains("fallback", case=False, na=False).mean())
            lines.append(f"glm_fallback_row_ratio(lambda_method contains 'fallback'): {fallback_ratio:.6f}")

    lines.append("")
    return "\n".join(lines)


def _build_accuracy_block(method: str, table: pd.DataFrame, split_label: str) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"METHOD METRICS: {method} (split={split_label})")
    lines.append("=" * 100)

    if "lambda_poisson" not in table.columns:
        lines.append("No baseline lambda_poisson found; skip accuracy metrics.")
        lines.append("")
        return "\n".join(lines)

    pred = pd.to_numeric(table.get("lambda_final"), errors="coerce")
    truth = pd.to_numeric(table.get("lambda_poisson"), errors="coerce")
    valid = pd.DataFrame({"pred": pred, "truth": truth})
    valid = valid[np.isfinite(valid["pred"]) & np.isfinite(valid["truth"]) & (valid["pred"] > 0) & (valid["truth"] > 0)]

    if valid.empty:
        lines.append("No valid rows for accuracy metrics.")
        lines.append("")
        return "\n".join(lines)

    err = valid["pred"] - valid["truth"]
    abs_pct_err = (err.abs() / valid["truth"].clip(lower=1e-12))

    mae = float(err.abs().mean())
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mape = float(abs_pct_err.mean())
    within_20 = float((abs_pct_err <= 0.20).mean())
    within_50 = float((abs_pct_err <= 0.50).mean())

    ttl_seconds = valid["pred"].apply(lambda x: ttl_from_lambda(float(x), target_freshness=TARGET_FRESHNESS))
    achieved_freshness = np.exp(-valid["truth"] * (ttl_seconds / 3600.0))
    freshness_hit = float((achieved_freshness >= TARGET_FRESHNESS).mean())

    lines.append(f"n_eval_rows: {len(valid)}")
    lines.append(f"mae_lambda: {mae:.8f}")
    lines.append(f"rmse_lambda: {rmse:.8f}")
    lines.append(f"mape_lambda: {mape:.8f}")
    lines.append(f"within_20pct_error: {within_20:.4f}")
    lines.append(f"within_50pct_error: {within_50:.4f}")
    lines.append(f"avg_achieved_freshness_at_pred_ttl: {float(achieved_freshness.mean()):.6f}")
    lines.append(f"freshness_target_hit_rate(target={TARGET_FRESHNESS:.2f}): {freshness_hit:.4f}")
    lines.append("")
    return "\n".join(lines)


def run_lambda_distribution_test(
    start_date: str = "2026-02-07",
    end_date: str = "2026-02-07",
    max_requests: int = 3000,
    min_intervals: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    processor = DataPipelineProcessor(data_root=str(ROOT_DIR / "data" / "cleaned_partitioned"))

    source_df, _ = processor.process(
        start_date=start_date,
        end_date=end_date,
        max_requests=max_requests,
    )

    pricing_source_df = processor._process_rates_and_prices(source_df.copy())
    pricing_source_df = processor._compute_market_and_price_change(pricing_source_df)

    return source_df, pricing_source_df


def build_report(
    start_date: str,
    end_date: str,
    max_requests: int,
    min_intervals: int,
    source_df: pd.DataFrame,
    pricing_source_df: pd.DataFrame,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("LAMBDA MODEL DISTRIBUTION TEST")
    lines.append("=" * 100)
    lines.append(f"Run Time: {now}")
    lines.append(f"Start Date: {start_date}")
    lines.append(f"End Date: {end_date}")
    lines.append(f"Max Requests: {max_requests}")
    lines.append(f"Min Intervals: {min_intervals}")
    lines.append(f"Loaded source_df rows: {len(source_df)}")
    lines.append(f"Pricing source rows (exploded): {len(pricing_source_df)}")
    lines.append(f"Unique cache_key in source_df: {source_df['cache_key'].nunique()}")
    lines.append("")

    for method in METHODS:
        if method == "glm":
            table = build_lambda_table(
                df_enriched=pricing_source_df,
                min_intervals=min_intervals,
                method=method,
                enable_train_test_split=True,
                train_ratio=0.8,
            )
        else:
            table = build_lambda_table(
                df_enriched=pricing_source_df,
                min_intervals=min_intervals,
                method=method,
            )
        if method == "glm":
            split_enabled = bool(table.attrs.get("train_test_split_enabled", False))
            lines.append("=" * 100)
            lines.append("GLM TRAIN/TEST METADATA")
            lines.append("=" * 100)
            lines.append(f"train_test_split_enabled: {split_enabled}")
            lines.append(f"train_ratio: {table.attrs.get('train_ratio')}")
            lines.append(f"split_timestamp_utc: {table.attrs.get('split_timestamp_utc')}")
            lines.append(f"n_rows_train: {table.attrs.get('n_rows_train')}")
            lines.append(f"n_rows_test: {table.attrs.get('n_rows_test')}")
            lines.append(f"glm_fitted_family: {table.attrs.get('glm_fitted_family')}")
            lines.append(f"fit_min_intervals: {table.attrs.get('fit_min_intervals')}")
            lines.append(f"lookup_min_intervals: {table.attrs.get('lookup_min_intervals')}")
            glm_test_metrics = table.attrs.get("glm_test_metrics", {})
            if glm_test_metrics:
                lines.append("glm_test_metrics:")
                for k, v in glm_test_metrics.items():
                    if isinstance(v, float):
                        lines.append(f"  {k}: {v:.8f}")
                    else:
                        lines.append(f"  {k}: {v}")
            else:
                lines.append("glm_test_metrics: {}")
            lines.append("")

            if "dataset_split" in table.columns:
                for split_name in ["train", "test"]:
                    split_df = table[table["dataset_split"] == split_name].copy()
                    if not split_df.empty:
                        lines.append(_build_method_block(method, split_df, split_label=split_name))
                        lines.append(_build_lambda_diagnostic_block(method, split_df, split_label=split_name))
                        lines.append(_build_accuracy_block(method, split_df, split_label=split_name))
            else:
                lines.append(_build_method_block(method, table, split_label="all"))
                lines.append(_build_lambda_diagnostic_block(method, table, split_label="all"))
                lines.append(_build_accuracy_block(method, table, split_label="all"))
        else:
            lines.append(_build_method_block(method, table, split_label="all"))
            lines.append(_build_lambda_diagnostic_block(method, table, split_label="all"))
            lines.append(_build_accuracy_block(method, table, split_label="all"))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lambda_model distribution test and export txt report.")
    parser.add_argument("--start-date", default="2026-02-07", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-02-07", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-requests", type=int, default=3000, help="Max request rows to sample")
    parser.add_argument("--min-intervals", type=int, default=5, help="Min intervals for lambda table building")
    parser.add_argument("--output-path", default=None, help="Optional output txt path")
    args = parser.parse_args()

    source_df, pricing_source_df = run_lambda_distribution_test(
        start_date=args.start_date,
        end_date=args.end_date,
        max_requests=args.max_requests,
        min_intervals=args.min_intervals,
    )

    report = build_report(
        start_date=args.start_date,
        end_date=args.end_date,
        max_requests=args.max_requests,
        min_intervals=args.min_intervals,
        source_df=source_df,
        pricing_source_df=pricing_source_df,
    )

    if args.output_path:
        output_path = (ROOT_DIR / args.output_path).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = ROOT_DIR / "Lambda" / f"lambda_distribution_{args.start_date}_{args.max_requests}_{ts}.txt"

    output_path.write_text(report, encoding="utf-8")
    print(f"Lambda distribution report written to: {output_path}")


if __name__ == "__main__":
    main()
