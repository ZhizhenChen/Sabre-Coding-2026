"""
Test script for demand score distribution.
Loads 10,000 requests from 2026-02-07 and writes distribution results to a txt file.
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add project roots to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_processing.pipeline import DataPipelineProcessor
from demand_forecasting.model_input import DemandScoreGenerator


def _format_series(series: pd.Series, float_digits: int = 6) -> str:
    """Format a Series as key-value text lines."""
    lines = []
    for idx, val in series.items():
        if isinstance(val, (float, np.floating)):
            lines.append(f"  {idx}: {val:.{float_digits}f}")
        else:
            lines.append(f"  {idx}: {val}")
    return "\n".join(lines)


def run_distribution_test(
    start_date: str = "2026-02-07",
    end_date: str = "2026-02-07",
    max_requests: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and generate demand scores for distribution analysis."""
    processor = DataPipelineProcessor(data_root=str(ROOT_DIR / "data" / "cleaned_partitioned"))

    source_df, prepared_df = processor.process(
        start_date=start_date,
        end_date=end_date,
        max_requests=max_requests,
    )

    model_generator = DemandScoreGenerator()
    score_df = model_generator.generate_demand_scores(
        processed_df=prepared_df,
        source_df=source_df,
        num_samples=None,
        output_parquet=None,
    )

    return source_df, prepared_df, score_df


def build_report(
    source_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    score_df: pd.DataFrame,
    date_label: str,
    requested_sample_label: str,
) -> str:
    """Build a txt report for demand score distributions."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    p = score_df["p_reuse"].astype(float)
    p_desc = p.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    score_cols = ["score_ap", "score_location", "score_duration", "score_rating_loc"]

    # Fixed bins to see distribution shape clearly.
    bins = [0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95, 0.99, 1.0]
    bin_labels = [
        "[0.00,0.01)",
        "[0.01,0.05)",
        "[0.05,0.10)",
        "[0.10,0.20)",
        "[0.20,0.40)",
        "[0.40,0.60)",
        "[0.60,0.80)",
        "[0.80,0.95)",
        "[0.95,0.99)",
        "[0.99,1.00]",
    ]
    p_bins = pd.cut(p.clip(0.0, 1.0), bins=bins, labels=bin_labels, include_lowest=True, right=True)
    p_bin_counts = p_bins.value_counts().reindex(bin_labels, fill_value=0)
    p_bin_pct = (p_bin_counts / len(p)).fillna(0.0)

    threshold_levels = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90]
    threshold_lines = []
    for th in threshold_levels:
        cnt = int((p >= th).sum())
        pct = 100.0 * cnt / len(p) if len(p) else 0.0
        threshold_lines.append(f"  p_reuse >= {th:>4.2f}: {cnt:6d} ({pct:6.2f}%)")

    lines = []
    lines.append("=" * 100)
    lines.append("DEMAND SCORE DISTRIBUTION TEST")
    lines.append("=" * 100)
    lines.append(f"Run Time: {now}")
    lines.append(f"Data Date: {date_label}")
    lines.append(f"Requested Sample Size: {requested_sample_label}")
    lines.append(f"Loaded source_df rows: {len(source_df)}")
    lines.append(f"Loaded prepared_df rows: {len(prepared_df)}")
    lines.append(f"Generated score_df rows: {len(score_df)}")
    lines.append(f"Unique cache_key in source_df: {source_df['cache_key'].nunique()}")
    lines.append(f"Unique cache_key in score_df: {score_df['cache_key'].nunique()}")
    lines.append("")

    lines.append("-" * 100)
    lines.append("P_REUSE SUMMARY STATISTICS")
    lines.append("-" * 100)
    lines.append(_format_series(p_desc, float_digits=8))
    lines.append("")

    lines.append("-" * 100)
    lines.append("P_REUSE BIN DISTRIBUTION")
    lines.append("-" * 100)
    for label in bin_labels:
        cnt = int(p_bin_counts[label])
        pct = 100.0 * float(p_bin_pct[label])
        lines.append(f"  {label:12s}: {cnt:6d} ({pct:6.2f}%)")
    lines.append("")

    lines.append("-" * 100)
    lines.append("P_REUSE THRESHOLD COVERAGE")
    lines.append("-" * 100)
    lines.extend(threshold_lines)
    lines.append("")

    lines.append("-" * 100)
    lines.append("COMPONENT SCORE SUMMARY")
    lines.append("-" * 100)
    for col in score_cols:
        col_desc = score_df[col].astype(float).describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
        lines.append(f"{col}:")
        lines.append(_format_series(col_desc, float_digits=8))
        lines.append("")

    lines.append("-" * 100)
    lines.append("TOP 20 HIGHEST P_REUSE")
    lines.append("-" * 100)
    top20 = score_df.nlargest(20, "p_reuse")
    for _, row in top20.iterrows():
        lines.append(
            "  "
            + f"cache_key={row['cache_key']}, p_reuse={float(row['p_reuse']):.8f}, "
            + f"ap={float(row['score_ap']):.6f}, loc={float(row['score_location']):.6f}, "
            + f"dur={float(row['score_duration']):.6f}, rating_loc={float(row['score_rating_loc']):.6f}"
        )
    lines.append("")

    lines.append("-" * 100)
    lines.append("TOP 20 LOWEST P_REUSE")
    lines.append("-" * 100)
    bottom20 = score_df.nsmallest(20, "p_reuse")
    for _, row in bottom20.iterrows():
        lines.append(
            "  "
            + f"cache_key={row['cache_key']}, p_reuse={float(row['p_reuse']):.8f}, "
            + f"ap={float(row['score_ap']):.6f}, loc={float(row['score_location']):.6f}, "
            + f"dur={float(row['score_duration']):.6f}, rating_loc={float(row['score_rating_loc']):.6f}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    start_date = "2026-01-07"
    end_date = "2026-02-07"
    date_label = f"{start_date}_to_{end_date}"
    max_requests = None
    requested_sample_label = "ALL"
    source_df, prepared_df, score_df = run_distribution_test(
        start_date=start_date,
        end_date=end_date,
        max_requests=max_requests,
    )

    report = build_report(
        source_df,
        prepared_df,
        score_df,
        date_label=date_label,
        requested_sample_label=requested_sample_label,
    )

    output_name = f"test_demand_score_distribution_{date_label}_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    output_path = ROOT_DIR / "Lambda" / output_name
    output_path.write_text(report, encoding="utf-8")

    print(f"Demand score distribution report written to: {output_path}")


if __name__ == "__main__":
    main()
