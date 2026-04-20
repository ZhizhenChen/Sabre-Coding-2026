from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Cache_System_Workflow.cache_pipeline import (
    _build_admission_lambda_state,
    _build_ttl_lookup,
    _prepare_requests,
)
from data_processing.pipeline import DataPipelineProcessor
from demand_forecasting.model_input import DemandScoreGenerator


def run_demand_lambda_correlation_test(
    start_date: str,
    end_date: str,
    max_requests: int,
    ttl_method: str,
    output_png: str | None,
    output_txt: str | None,
    plot_sample: int,
) -> tuple[Path, Path]:
    root_dir = ROOT_DIR

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_png is None:
        output_png = f"demand_lambda_corr_{start_date}_to_{end_date}_{max_requests}_{ttl_method}_{ts}.png"
    if output_txt is None:
        output_txt = f"demand_lambda_corr_{start_date}_to_{end_date}_{max_requests}_{ttl_method}_{ts}.txt"

    output_png_path = root_dir / output_png
    output_txt_path = root_dir / output_txt

    processor = DataPipelineProcessor(data_root=str(root_dir / "data" / "cleaned_partitioned"))
    source_df, prepared_df = processor.process(
        start_date=start_date,
        end_date=end_date,
        max_requests=max_requests,
    )

    pricing_source_df = processor._process_rates_and_prices(source_df.copy())
    pricing_source_df = processor._compute_market_and_price_change(pricing_source_df)

    model_generator = DemandScoreGenerator()
    p_reuse_df = model_generator.generate_demand_scores(
        processed_df=prepared_df,
        source_df=source_df,
        num_samples=None,
        output_parquet=None,
    )

    ttl_lookup = _build_ttl_lookup(pricing_source_df, ttl_method=ttl_method)
    admission_state = _build_admission_lambda_state(pricing_source_df, ttl_method=ttl_method)
    prepared_requests = _prepare_requests(
        source_df,
        p_reuse_df,
        ttl_lookup_by_bucket=ttl_lookup,
        admission_state=admission_state,
    )

    corr_df = pd.DataFrame(
        {
            "cache_key": [x.request.cache_key() for x in prepared_requests],
            "p_reuse": [float(x.p_reuse) for x in prepared_requests],
            "lambda_i": [float(x.lambda_i) for x in prepared_requests],
        }
    )
    corr_df = corr_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["p_reuse", "lambda_i"]).copy()
    corr_df = corr_df[corr_df["lambda_i"] > 0].copy()

    pearson = float(corr_df["p_reuse"].corr(corr_df["lambda_i"], method="pearson"))
    spearman = float(corr_df["p_reuse"].corr(corr_df["lambda_i"], method="spearman"))

    plot_df = corr_df
    if plot_sample > 0 and len(corr_df) > plot_sample:
        plot_df = corr_df.sample(n=plot_sample, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hb1 = axes[0].hexbin(
        plot_df["p_reuse"],
        plot_df["lambda_i"],
        gridsize=45,
        mincnt=1,
        cmap="Blues",
    )
    axes[0].set_title("Demand vs Lambda (Raw)")
    axes[0].set_xlabel("p_reuse")
    axes[0].set_ylabel("lambda_i")
    fig.colorbar(hb1, ax=axes[0], label="count")

    hb2 = axes[1].hexbin(
        plot_df["p_reuse"],
        np.log1p(plot_df["lambda_i"]),
        gridsize=45,
        mincnt=1,
        cmap="Greens",
    )
    axes[1].set_title("Demand vs log(1 + Lambda)")
    axes[1].set_xlabel("p_reuse")
    axes[1].set_ylabel("log(1 + lambda_i)")
    fig.colorbar(hb2, ax=axes[1], label="count")

    fig.suptitle(
        f"Correlation ({ttl_method}) | pearson={pearson:.4f}, spearman={spearman:.4f}",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(output_png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    lines = []
    lines.append("DEMAND-LAMBDA CORRELATION TEST")
    lines.append(f"start_date={start_date}")
    lines.append(f"end_date={end_date}")
    lines.append(f"max_requests={max_requests}")
    lines.append(f"ttl_method={ttl_method}")
    lines.append(f"n_rows={len(corr_df)}")
    lines.append(f"n_unique_keys={corr_df['cache_key'].nunique()}")
    lines.append(f"pearson_corr={pearson:.8f}")
    lines.append(f"spearman_corr={spearman:.8f}")
    lines.append("")

    p_desc = corr_df["p_reuse"].describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
    l_desc = corr_df["lambda_i"].describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])

    lines.append("[p_reuse_distribution]")
    for k, v in p_desc.items():
        lines.append(f"{k}: {float(v):.8f}")
    lines.append("")
    lines.append("[lambda_i_distribution]")
    for k, v in l_desc.items():
        lines.append(f"{k}: {float(v):.8f}")

    output_txt_path.write_text("\n".join(lines), encoding="utf-8")

    return output_png_path, output_txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Test correlation between demand score and lambda_i.")
    parser.add_argument("--start-date", type=str, default="2026-02-07")
    parser.add_argument("--end-date", type=str, default="2026-02-07")
    parser.add_argument("--max-requests", type=int, default=100000)
    parser.add_argument("--ttl-method", type=str, default="pp", choices=["pp", "glm", "km", "rule_based"])
    parser.add_argument("--output-png", type=str, default=None)
    parser.add_argument("--output-txt", type=str, default=None)
    parser.add_argument(
        "--plot-sample",
        type=int,
        default=50000,
        help="Max points to draw in chart (0 means draw all points).",
    )
    args = parser.parse_args()

    png_path, txt_path = run_demand_lambda_correlation_test(
        start_date=args.start_date,
        end_date=args.end_date,
        max_requests=args.max_requests,
        ttl_method=args.ttl_method,
        output_png=args.output_png,
        output_txt=args.output_txt,
        plot_sample=args.plot_sample,
    )

    print(f"Saved correlation plot: {png_path}")
    print(f"Saved summary txt: {txt_path}")


if __name__ == "__main__":
    main()
