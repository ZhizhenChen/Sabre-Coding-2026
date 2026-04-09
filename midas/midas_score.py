from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


LEAD_TIME_BREAKS = [1, 3, 7, 30]
LEAD_TIME_LABELS = ["same_day", "short", "mid", "long", "very_long"]


def _assign_lead_time_bucket(lead_time_days: int) -> str:
    for boundary, label in zip(LEAD_TIME_BREAKS, LEAD_TIME_LABELS[:-1]):
        if lead_time_days <= boundary:
            return label
    return LEAD_TIME_LABELS[-1]


def _resolve_user_series(frame: pd.DataFrame) -> pd.Series:
    if "rq_user" in frame.columns:
        return frame["rq_user"].astype(str).replace({"": "anonymous", "nan": "anonymous"}).fillna("anonymous")
    if "rq_correlation_id" in frame.columns:
        return frame["rq_correlation_id"].astype(str).replace({"": "anonymous", "nan": "anonymous"}).fillna("anonymous")
    return pd.Series(["anonymous"] * len(frame), index=frame.index, dtype="object")


def _resolve_state(frame: pd.DataFrame, state_col: Optional[str] = None) -> pd.Series:
    if state_col and state_col in frame.columns:
        return frame[state_col].astype(str).replace({"": "unknown", "nan": "unknown"}).fillna("unknown")

    if "lead_time_bucket" in frame.columns:
        lead_bucket = frame["lead_time_bucket"].astype(str)
    else:
        lead = pd.to_numeric(frame.get("lead_time", 0), errors="coerce").fillna(0).clip(lower=0).astype(int)
        lead_bucket = lead.apply(_assign_lead_time_bucket)

    hotel_grouped = None
    for candidate in ["hotel_grouped", "hotel_code", "chain_code"]:
        if candidate in frame.columns:
            hotel_grouped = frame[candidate].astype(str)
            break
    if hotel_grouped is None:
        hotel_grouped = pd.Series(["unknown"] * len(frame), index=frame.index, dtype="object")

    rate_source = frame.get("rate_source")
    if rate_source is None:
        rate_source = pd.Series(["unknown"] * len(frame), index=frame.index, dtype="object")
    rate_source = rate_source.astype(str)

    return (hotel_grouped + "|" + rate_source + "|" + lead_bucket).replace({"": "unknown", "nan": "unknown"}).fillna("unknown")


def _markov_event_scores(events: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["cache_key", "markov_state_score"])

    alpha = float(max(alpha, 1e-6))
    states = sorted(events["state"].dropna().astype(str).unique().tolist())
    n_states = max(1, len(states))

    prior_counts = events["state"].value_counts().to_dict()
    prior_total = float(sum(prior_counts.values()))
    prior_prob = {s: (float(prior_counts.get(s, 0.0)) + alpha) / (prior_total + alpha * n_states) for s in states}

    pair_counts: dict[tuple[str, str], int] = {}
    prev_out_counts: dict[str, int] = {}
    for _, g in events.groupby("user_id", dropna=False):
        seq = g["state"].astype(str).tolist()
        for i in range(1, len(seq)):
            prev_s, cur_s = seq[i - 1], seq[i]
            pair_counts[(prev_s, cur_s)] = pair_counts.get((prev_s, cur_s), 0) + 1
            prev_out_counts[prev_s] = prev_out_counts.get(prev_s, 0) + 1

    rows = []
    for _, g in events.groupby("user_id", dropna=False):
        g = g.sort_values("rq_timestamp")
        prev_state = None
        for rec in g.itertuples(index=False):
            cur_state = str(rec.state)
            if prev_state is None:
                sc = float(prior_prob.get(cur_state, 1.0 / n_states))
            else:
                num = float(pair_counts.get((prev_state, cur_state), 0)) + alpha
                den = float(prev_out_counts.get(prev_state, 0)) + alpha * n_states
                sc = num / den if den > 0 else (1.0 / n_states)
            rows.append(
                {
                    "cache_key": str(rec.cache_key),
                    "markov_event_score": float(np.clip(sc, 0.0, 1.0)),
                }
            )
            prev_state = cur_state

    scored = pd.DataFrame(rows)
    out = scored.groupby("cache_key", dropna=False)["markov_event_score"].mean().reset_index(name="markov_state_score")
    return out


def build_midas_scores(
    requests_df: pd.DataFrame,
    p_reuse_df: pd.DataFrame,
    w_demand: float = 0.8,
    w_markov: float = 0.2,
    alpha: float = 1.0,
    state_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build key-level MIDAS score = demand score + Markov state uplift.

    Returns columns: cache_key, p_reuse, markov_state_score, midas_score, midas_version.
    """
    if "cache_key" not in requests_df.columns:
        raise ValueError("requests_df must contain 'cache_key'")

    p_cols = {"cache_key", "p_reuse"}
    if not p_cols.issubset(set(p_reuse_df.columns)):
        raise ValueError("p_reuse_df must contain 'cache_key' and 'p_reuse'")

    base = p_reuse_df[["cache_key", "p_reuse"]].copy()
    base["cache_key"] = base["cache_key"].astype(str)
    base["p_reuse"] = pd.to_numeric(base["p_reuse"], errors="coerce")
    base = base.groupby("cache_key", as_index=False)["p_reuse"].mean()

    events = requests_df.copy()
    if "rq_timestamp" not in events.columns:
        raise ValueError("requests_df must contain 'rq_timestamp'")
    events["rq_timestamp"] = pd.to_datetime(events["rq_timestamp"], errors="coerce", utc=True)
    events = events.dropna(subset=["rq_timestamp", "cache_key"]).copy()

    events["cache_key"] = events["cache_key"].astype(str)
    events["user_id"] = _resolve_user_series(events)
    events["state"] = _resolve_state(events, state_col=state_col)
    events = events.sort_values(["user_id", "rq_timestamp"], kind="stable").reset_index(drop=True)

    markov = _markov_event_scores(events, alpha=alpha)

    out = base.merge(markov, on="cache_key", how="left")
    global_markov = float(markov["markov_state_score"].mean()) if not markov.empty else 0.5
    out["markov_state_score"] = pd.to_numeric(out["markov_state_score"], errors="coerce").fillna(global_markov).clip(0.0, 1.0)
    out["p_reuse"] = pd.to_numeric(out["p_reuse"], errors="coerce").fillna(0.5).clip(0.0, 1.0)

    wd = max(0.0, float(w_demand))
    wm = max(0.0, float(w_markov))
    if wd + wm <= 0:
        wd, wm = 0.8, 0.2
    z = wd + wm
    wd, wm = wd / z, wm / z

    out["midas_score"] = (wd * out["p_reuse"] + wm * out["markov_state_score"]).clip(0.0, 1.0)
    out["midas_version"] = "MIDAS_v1"
    return out[["cache_key", "p_reuse", "markov_state_score", "midas_score", "midas_version"]]
