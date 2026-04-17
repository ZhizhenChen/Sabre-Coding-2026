from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lambda_model import build_lambda_table


def _format_overall_line(lambda_table: pd.DataFrame) -> str:
    if lambda_table.empty:
        return "overall_lambda_final(weighted_by_exposure)=NA"
    w = lambda_table["exposure_hours"].to_numpy(dtype=float)
    v = lambda_table["lambda_final"].to_numpy(dtype=float)
    weighted = float(np.average(v, weights=w)) if np.sum(w) > 0 else float(np.mean(v))
    return f"overall_lambda_final(weighted_by_exposure)={weighted:.8f}"


def _format_method_summary(method: str, lambda_table: pd.DataFrame) -> str:
    if lambda_table.empty:
        return f"{method}: groups=0 overall_lambda=NA"
    return (
        f"{method}: groups={len(lambda_table)} "
        f"overall_lambda={_format_overall_line(lambda_table).split('=')[1]} "
        f"median_lambda={lambda_table['lambda_final'].median():.8f} "
        f"mean_lambda={lambda_table['lambda_final'].mean():.8f}"
    )


def _extract_failure_reason(lambda_table: pd.DataFrame) -> str:
    if lambda_table.empty or "lambda_debug_reason" not in lambda_table.columns:
        return ""
    reasons = [str(v) for v in lambda_table["lambda_debug_reason"].dropna().unique().tolist() if str(v).strip()]
    return reasons[0] if reasons else ""


def _to_lines(df: pd.DataFrame, max_rows: int = 20) -> Iterable[str]:
    if df.empty:
        yield "No groups passed min_intervals filter."
        return
    cols = [
        "hotel_code",
        "lead_time",
        "rate_source",
        "n_intervals",
        "n_events",
        "exposure_hours",
        "lambda_method",
        "lambda_final",
    ]
    show = df[cols].head(max_rows)
    yield show.to_string(index=False)


def _save_lambda_distribution_plot(results: dict[str, pd.DataFrame], output_path: str) -> None:
    methods = list(results.keys())
    if not methods:
        return

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5), squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        table = results[method]
        if table.empty:
            ax.text(0.5, 0.5, f"{method}\nno groups", ha="center", va="center")
            ax.set_axis_off()
            continue

        values = table["lambda_final"].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            ax.text(0.5, 0.5, f"{method}\nno valid lambda values", ha="center", va="center")
            ax.set_axis_off()
            continue

        ax.hist(values, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_title(f"{method.upper()} lambda distribution\n(n={len(values)})")
        ax.set_xlabel("lambda_final")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.98,
            0.97,
            f"μ={values.mean():.3f}\nσ={values.std():.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_smoke_km_lambda(
    processed_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    partition_date: Optional[str] = "2026-02-07",
    max_requests: int = 3000,
    output_path: str = "lambda_methods_2026-02-07_smoke.txt",
) -> None:
    processed_df = processed_df.copy().head(max_requests)

    if "rq_timestamp" in processed_df.columns:
        ts = pd.to_datetime(processed_df["rq_timestamp"], errors="coerce", utc=True)
        if start_date and end_date:
            start_ts = pd.Timestamp(start_date, tz="UTC")
            end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            processed_df = processed_df[(ts >= start_ts) & (ts <= end_ts)].copy()
        elif partition_date:
            processed_df = processed_df[ts.dt.strftime("%Y-%m-%d") == partition_date].copy()

    required = ["hotel_code", "lead_time", "rate_source", "rq_timestamp", "price_change"]
    missing = [c for c in required if c not in processed_df.columns]
    if missing:
        raise ValueError(f"Input df is missing required columns: {missing}")

    processed_df["rq_timestamp"] = pd.to_datetime(processed_df["rq_timestamp"], errors="coerce", utc=True)
    processed_df = processed_df.dropna(subset=["rq_timestamp", "hotel_code", "lead_time", "rate_source"]).copy()

    grouped_stats = (
        processed_df.groupby(["hotel_code", "lead_time", "rate_source"], dropna=False)
        .agg(obs_count=("price_change", "size"), price_change_count=("price_change", "sum"))
        .reset_index()
    )
    method_specs = [
        ("poisson", 2),
        ("glm", 2),
        ("fallback", 1),
    ]

    lambda_tables: dict[str, pd.DataFrame] = {}
    for method, min_intervals in method_specs:
        lambda_tables[method] = build_lambda_table(
            processed_df,
            min_intervals=min_intervals,
            method=method,
        )

    lines = [
        "Lambda Methods Smoke Run",
        f"start_date={start_date if start_date else 'ALL'}",
        f"end_date={end_date if end_date else 'ALL'}",
        f"partition_date={partition_date if partition_date else 'ALL'}",
        f"processed_rows={len(processed_df)}",
        f"sample_rows={max_requests}",
        f"grouped_stats_rows={len(grouped_stats)}",
        f"methods={', '.join(method for method, _ in method_specs)}",
        "",
    ]

    for method, _ in method_specs:
        table = lambda_tables[method]
        lines.append(f"=== {method.upper()} ===")
        lines.append(f"lambda_groups={len(table)}")
        lines.append(_format_method_summary(method, table))
        failure_reason = _extract_failure_reason(table)
        if failure_reason:
            lines.append(f"failure_reason={failure_reason}")
        lines.append("Top groups by intervals:")
        lines.extend(_to_lines(table, max_rows=20))
        lines.append("")

    txt_path = Path(output_path)
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    plot_path = txt_path.with_suffix(".png")
    _save_lambda_distribution_plot(lambda_tables, str(plot_path))
    lines.append(f"saved_plot={plot_path}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved text summary to {txt_path}")
    print(f"Saved distribution plot to {plot_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run non-KM lambda smoke test from a prepared dataframe parquet.")
    parser.add_argument("--input-parquet", required=True, help="Parquet file containing processed dataframe")
    parser.add_argument("--start-date", default=None, help="Optional start date (YYYY-MM-DD) for filtering")
    parser.add_argument("--end-date", default=None, help="Optional end date (YYYY-MM-DD) for filtering")
    parser.add_argument("--partition-date", default="2026-02-07")
    parser.add_argument("--max-requests", type=int, default=3000)
    parser.add_argument("--output-path", default="lambda_methods_2026-02-07_smoke.txt")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    processed_df = pd.read_parquet(args.input_parquet)
    run_smoke_km_lambda(
        processed_df=processed_df,
        start_date=args.start_date,
        end_date=args.end_date,
        partition_date=args.partition_date,
        max_requests=args.max_requests,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
