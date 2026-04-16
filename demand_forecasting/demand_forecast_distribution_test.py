from __future__ import annotations

import argparse
from pathlib import Path

from data_processing.pipeline import DataPipelineProcessor
from demand_forecasting.model_input import DemandScoreGenerator


def build_summary_lines(
    start_date: str,
    end_date: str,
    max_requests: int,
    source_rows: int,
    prepared_rows: int,
    p_reuse_df,
) -> list[str]:
    s = p_reuse_df["p_reuse"]

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("DEMAND FORECASTER SCORE DISTRIBUTION")
    lines.append("=" * 70)
    lines.append(f"start_date={start_date}")
    lines.append(f"end_date={end_date}")
    lines.append(f"max_requests={max_requests}")
    lines.append(f"source_rows={source_rows}")
    lines.append(f"prepared_rows={prepared_rows}")
    lines.append(f"p_reuse_rows={len(p_reuse_df)}")
    lines.append("")

    lines.append("p_reuse stats:")
    lines.append(f"  mean={s.mean():.6f}")
    lines.append(f"  median={s.median():.6f}")
    lines.append(f"  std={s.std():.6f}")
    lines.append(f"  min={s.min():.6f}")
    lines.append(f"  max={s.max():.6f}")

    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        lines.append(f"  q{int(q * 100):02d}={s.quantile(q):.6f}")

    component_cols = ["score_ap", "score_location", "score_duration", "score_rating_loc"]
    for col in component_cols:
        if col in p_reuse_df.columns:
            c = p_reuse_df[col]
            lines.append("")
            lines.append(f"{col} stats:")
            lines.append(f"  mean={c.mean():.6f}")
            lines.append(f"  median={c.median():.6f}")
            lines.append(f"  std={c.std():.6f}")
            lines.append(f"  min={c.min():.6f}")
            lines.append(f"  max={c.max():.6f}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demand forecaster score distribution test.")
    parser.add_argument("--start-date", default="2026-02-07", help="Start date, e.g. 2026-02-07")
    parser.add_argument("--end-date", default="2026-02-07", help="End date, e.g. 2026-02-07")
    parser.add_argument("--max-requests", type=int, default=3000, help="Max raw requests to keep")
    parser.add_argument(
        "--data-root",
        default="data/cleaned_partitioned",
        help="Path to cleaned partitioned parquet root",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output txt file path (default: demand_forecaster_distribution_<start>_<max>.txt)",
    )
    args = parser.parse_args()

    root_dir = Path(".").resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root_dir / data_root

    processor = DataPipelineProcessor(data_root=str(data_root))
    source_df, prepared_df = processor.process(
        start_date=args.start_date,
        end_date=args.end_date,
        max_requests=args.max_requests,
    )

    model_generator = DemandScoreGenerator()
    p_reuse_df = model_generator.generate_demand_scores(
        processed_df=prepared_df,
        source_df=source_df,
        num_samples=None,
        output_parquet=None,
    )

    output_path = Path(args.output) if args.output else root_dir / (
        f"demand_forecaster_distribution_{args.start_date}_{args.max_requests}.txt"
    )

    lines = build_summary_lines(
        start_date=args.start_date,
        end_date=args.end_date,
        max_requests=args.max_requests,
        source_rows=len(source_df),
        prepared_rows=len(prepared_df),
        p_reuse_df=p_reuse_df,
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {output_path}")
    print("--- Preview ---")
    print("\n".join(lines[:24]))


if __name__ == "__main__":
    main()
