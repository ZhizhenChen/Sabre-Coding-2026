# Sabre Hotel Cache Evaluation

This repository runs end-to-end cache policy experiments for hotel pricing requests.

Main objectives:
- generate demand reuse score (`p_reuse`)
- optionally build MIDAS-v2 admission score (`midas_score`) using latent behavior states + Markov smoothing
- estimate TTL by multiple methods (`glm`, `pp`, `rule_based`, `km`)
- evaluate two-tier cache workflow vs LRU baseline
- report hit rate, provider calls, staleness, prewarm quality, and eviction behavior

## Current Entry Point

Use [Cache_System_Workflow/cache_pipeline.py](Cache_System_Workflow/cache_pipeline.py) as the main experiment script.

Current setup in this script:
- admission score source is configurable: `p_reuse` or `midas`
- TTL method is varied across `glm`, `pp`, `rule_based`, `km`
- workflow engine is [Cache_System_Workflow/sabre_cache_workflow_v2.py](Cache_System_Workflow/sabre_cache_workflow_v2.py)
- MIDAS module is [midas/midas_score.py](midas/midas_score.py)

## Repository Structure

- [data_processing/pipeline.py](data_processing/pipeline.py): data preprocessing and feature engineering
- [demand_forecasting/model_input.py](demand_forecasting/model_input.py): demand scoring (`p_reuse`)
- [midas/midas_score.py](midas/midas_score.py): latent behavior intent scoring + MIDAS-v2 correction
- [Lambda/lambda_model.py](Lambda/lambda_model.py): lambda/KM/GLM estimation utilities
- [Cache_System_Workflow/cache_pipeline.py](Cache_System_Workflow/cache_pipeline.py): TTL method evaluation driver
- [Cache_System_Workflow/sabre_cache_workflow_v2.py](Cache_System_Workflow/sabre_cache_workflow_v2.py): two-tier cache implementation
- [LRU/simulate_lru_baseline.py](LRU/simulate_lru_baseline.py): LRU baseline cache
- [run_workflow.sh](run_workflow.sh): shell wrapper for workflow execution

## Data Flow

1. Load partition data with `DataPipelineProcessor.process(partition_date, max_rows)`.
2. Produce:
   - `source_df`: enriched row-level frame (includes exploded rate/source rows)
   - `prepared_df`: hourly feature frame for demand model
3. Generate `p_reuse` from `prepared_df` and `source_df`.
4. Optionally build `midas_score` from latent behavior-state Markov model.
5. Build request records (`PreparedWorkflowInput`) for cache simulation.
6. Run cache workflow and LRU baseline.
7. Write text report.

## Payload Format (Current)

Provider payload is request-keyed and source-aware:

```json
{
  "request_key": "...",
  "offers": [
    {"source": "source1", "price": 100.0},
    {"source": "source2", "price": 120.0}
  ]
}
```

`offers` are built from `source_df` by grouping per `cache_key` and `rate_source`.

## Metrics

Report includes:
- `hit_rate`
- `provider_total_calls`
- `api_call_reduction_pct_vs_lru`
- `stale_rate_served_pct`
- `prewarm_precision`
- `eviction_count`
- `eviction_accuracy`
- `avg_query_after_eviction`

Hit definition:
- a request is counted as hit when workflow result source contains `"hit"` (`controlled_hit` or `uncontrolled_hit`)

## Environment Setup

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas pyarrow matplotlib xgboost statsmodels
```

If XGBoost on macOS raises OpenMP errors:

```bash
brew install libomp
```

If using demand checkpoints in `demand_forecasting/checkpoint_model_*`, install:

```bash
pip install gluonts[torch]
```

## Run

From repo root, run the main evaluator:

```bash
python Cache_System_Workflow/cache_pipeline.py
```

Custom run example:

```bash
python Cache_System_Workflow/cache_pipeline.py \
  --partition-date 2026-02-07 \
  --max-requests 3000 \
  --output-path workflow_ttl_methods_eval_custom.txt \
  --controlled-capacity 100 \
  --uncontrolled-capacity 900 \
  --admission-source midas \
  --w-demand 0.85 \
  --w-intent 0.15 \
  --midas-eta 0.35 \
  --midas-tau 0.08 \
  --midas-alpha 1.0 \
  --score-percentile 0.7 \
  --prefetch-ratio 0.2
```

Shell wrapper:

```bash
./run_workflow.sh --help
```

## Notes

- `source_df` can be larger than request count because rate details are exploded by source.
- Stay-date invalidation is enabled in cache validity check: when current date is later than stay start date, cached entry is treated as invalid.
- `Cache_System_Workflow/sabre_cache_workflow.py` is legacy and ignored by git; use v2.
