# Lambda TTL Modeling

This folder contains lambda/TTL modeling logic for cache expiration in hotel pricing.

## Core Equations

Let `lambda` be the price-change intensity (changes per hour).

- Freshness probability after `t` hours:
  - `P_fresh(t) = exp(-lambda * t)`
- Stale probability after `t` hours:
  - `P_stale(t) = 1 - exp(-lambda * t)`

For target freshness `q` (`0 < q < 1`), TTL is:

- `TTL_hours = -ln(q) / lambda`
- `TTL_seconds = TTL_hours * 3600`

Special case used throughout this repo (`q=0.8`):

- `-ln(0.8) ~= 0.2231`
- `TTL_hours = 0.2231 / lambda`

So the core task is estimating `lambda` well.

---

## Data Preparation

Implemented in `lambda_model.py`:

- `bucket_lead_time(horizon)`
- `bucket_duration(duration)`
- `encode_hotel(df, min_count=4)`
- `_prepare_enriched_dataframe(df_enriched)`

Required columns:

- `rq_timestamp`, `hotel_code`, `lead_time`, `duration`, `rate_source`, `price_change`

Rules:

1. Invalid rows are removed:
- drop `lead_time < 0`
- drop `duration <= 0`

2. Lead-time buckets:
- `0 -> same_day`
- `1-3 -> 1_to_3d`
- `4-7 -> 4_to_7d`
- `8-14 -> 8_to_14d`
- `15-30 -> 15_to_30d`
- `31-90 -> 31_to_90d`
- `91+ -> 91plus`

3. Duration buckets:
- `1 -> 1_night`
- `2 -> 2_nights`
- `3 -> 3_nights`
- `4-5 -> 4_to_5n`
- `6-7 -> 6_to_7n`
- `8-14 -> 8_to_14n`
- `15+ -> 15plus`

4. Sparse hotel handling:
- hotels with `< 4` rows -> `"rare_hotel"`
- others keep original `hotel_code`

---

## Poisson Path

Main function:

- `build_lambda_table_poisson(df_enriched, min_intervals=5)`

Group key:

- `(hotel_encoded, duration_bucket, lead_time_bucket, rate_source)`

Per-group estimates:

- `total_changes = sum(price_change)`
- `total_exposure_hours = sum(delta_hours)`
- `lambda_poisson = total_changes / total_exposure_hours`
- `implied_ttl_poisson_hours = 0.2231 / lambda_poisson`

Global fallback:

- `global_lambda = sum(total_changes) / sum(total_exposure_hours)`

Serving:

- `serve_poisson_ttl(...)`
- lookup by 4-part key, fallback to `global_lambda` if missing/invalid

---

## GLM Path

Main function:

- `build_lambda_table_glm(df_enriched, min_intervals=5, use_negative_binomial=True, enable_train_test_split=False, train_ratio=0.8)`

Model form:

- Response: `Y_i = total_changes_i`
- Exposure: `E_i = total_exposure_hours_i`
- Offset: `log(E_i)`
- Linear predictor:
  - `log(mu_i) = beta0 + beta_hotel + beta_lead_time + beta_duration + beta_rate_source + log(E_i)`
- Therefore:
  - `mu_i = E[Y_i | x_i]`
  - `lambda_glm_i = mu_i / E_i`

Feature encoding:

- All predictors are categorical using treatment coding with selected references.

Reference-level selection:

- references are chosen to be closest to the training-set global change rate.

Family:

- Negative Binomial by default (`use_negative_binomial=True`), or Poisson.

Important extraction detail:

- `predicted_events = model.predict(...)` (already on response scale)
- `lambda_glm = predicted_events / total_exposure_hours`
- do not apply extra `exp(...)` to `predicted_events`

### GLM Train/Test Split

Default behavior is unchanged for pipeline compatibility:

- `enable_train_test_split=False` by default

When enabled:

- split by time (`rq_timestamp`) using quantile at `train_ratio`
- train: `rq_timestamp <= split_timestamp`
- test: `rq_timestamp > split_timestamp`

Model fitting and lookup:

- fit uses train groups only
- lookup dictionary (`lambda_lookup_dict_glm`) is built from train predictions only

Returned table (when split enabled):

- includes both `dataset_split="train"` and `dataset_split="test"` rows

Returned attrs include:

- `train_test_split_enabled`
- `train_ratio`
- `split_timestamp_utc`
- `n_rows_train`, `n_rows_test`
- `glm_test_metrics` (if computable)

GLM test metrics compare `lambda_glm` vs test-group empirical `lambda_poisson`:

- `MAE = mean(|lambda_glm - lambda_poisson|)`
- `RMSE = sqrt(mean((lambda_glm - lambda_poisson)^2))`
- `MAPE = mean(|lambda_glm - lambda_poisson| / max(lambda_poisson, 1e-12))`

Serving:

- `serve_glm_ttl(...)`
- fallback waterfall:
  1. full key `(hotel_encoded, duration_bucket, lead_time_bucket, rate_source)`
  2. `("rare_hotel", duration_bucket, lead_time_bucket, rate_source)`
  3. `("rare_hotel", ref_dur, lead_time_bucket, ref_rs)`
  4. `global_lambda`

---

## KM Path (Legacy)

Main function:

- `build_lambda_table_km(df_enriched, min_intervals=5)`

This path is kept for backward compatibility.

Core KM pieces:

- `_km_survival(...)`
- `_median_from_survival(...)`
- `_rmst_from_survival(...)`
- `estimate_lambda_km(...)`

### `_km_survival(durations, events)` definition

Inputs:

- `durations`: positive interval lengths (hours)
- `events`: event indicator per interval (`1` if price changed, `0` otherwise)

At each unique event/censor time `t_j`:

- `n_j`: number at risk just before `t_j`
- `d_j`: number of events at `t_j`

KM step update used by implementation:

- `S(t_j) = S(t_{j-1}) * (1 - d_j / n_j)`
- with `S(0) = 1`

Returned values:

- `times = [t_1, t_2, ...]` (sorted unique durations)
- `survival = [S(t_1), S(t_2), ...]`

Equivalent product form:

- `S(t) = product_{t_j <= t} (1 - d_j / n_j)`

From this KM curve:

- median survival time is the first `t` where `S(t) <= 0.5`
- RMST is the area under stepwise survival curve:
  - `RMST = integral_0^tau S(u) du`

KM-related rates:

- `lambda_empirical = n_events / total_exposure_hours`
- `lambda_km_median = ln(2) / median_hours` (when median finite)
- `lambda_km_rmst = 1 / rmst_hours` (when RMST finite)

In current table output, `lambda_final` for KM path uses `lambda_empirical`.

---

## Fallback Path

Main function:

- `build_lambda_table_fallback(df_enriched, min_intervals=1)`

Behavior:

- robust empirical lambda with global fill
- used when a method has insufficient data or fit failure

---

## Rule-Based TTL Baseline

`rule_based` is a workflow baseline policy and is not estimated in `Lambda/lambda_model.py`.

- `Lambda/lambda_model.py` covers data-driven methods: `km`, `poisson`, `glm`, `fallback`
- rule-based TTL mapping is defined in workflow code (`Cache_System_Workflow/cache_pipeline.py`)

---

## Unified Entry Point

- `build_lambda_table(df_enriched, min_intervals=5, method=...)`

Supported methods:

- `km`
- `poisson`
- `glm`
- `fallback`

GLM-only kwargs can be passed through this unified function, for example:

- `build_lambda_table(..., method="glm", enable_train_test_split=True, train_ratio=0.8)`

---

## Sanity Check Utility

- `print_sanity_check(lambda_poisson_table, lambda_glm_table, target_freshness=0.8)`

Prints:

- global lambda and implied TTL
- min/max lambda and implied TTL for Poisson + GLM
- warnings for suspicious ranges

---

## Minimal Usage Examples

```python
from Lambda.lambda_model import build_lambda_table

# 1) Standard GLM (no split)
glm_table = build_lambda_table(
    df_enriched=pricing_source_df,
    min_intervals=5,
    method="glm",
)

# 2) GLM with time-based train/test split
glm_split_table = build_lambda_table(
    df_enriched=pricing_source_df,
    min_intervals=5,
    method="glm",
    enable_train_test_split=True,
    train_ratio=0.8,
)

print(glm_split_table.attrs.get("split_timestamp_utc"))
print(glm_split_table.attrs.get("glm_test_metrics"))
```
