from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

STATES = [
    "urgent_repeat",
    "planned_repeat",
    "planned_explore",
    "price_probe",
    "cold_start",
]
STATE_TO_IDX = {state: i for i, state in enumerate(STATES)}
K_STATES = len(STATES)

# Higher value means stronger expected short-horizon reuse likelihood.
STATE_REUSE_WEIGHTS = np.array([1.00, 0.85, 0.45, 0.30, 0.20], dtype=float)


@dataclass(frozen=True)
class MidasParams:
    w_demand: float = 0.85
    w_intent: float = 0.15
    eta: float = 0.35
    tau: float = 0.08
    alpha: float = 1.0
    eps: float = 1e-4


def _resolve_user_series(frame: pd.DataFrame) -> pd.Series:
    for col in ["rq_user", "rq_correlation_id", "session_id", "user_id"]:
        if col in frame.columns:
            return frame[col].astype(str).replace({"": "anonymous", "nan": "anonymous"}).fillna("anonymous")
    return pd.Series(["anonymous"] * len(frame), index=frame.index, dtype="object")


def _resolve_hotel_series(frame: pd.DataFrame) -> pd.Series:
    for col in ["hotel_code", "hotel_grouped", "chain_code"]:
        if col in frame.columns:
            return frame[col].astype(str).replace({"": "unknown_hotel", "nan": "unknown_hotel"}).fillna("unknown_hotel")
    return pd.Series(["unknown_hotel"] * len(frame), index=frame.index, dtype="object")


def _resolve_lead_time(frame: pd.DataFrame) -> pd.Series:
    if "lead_time" in frame.columns:
        return pd.to_numeric(frame["lead_time"], errors="coerce").fillna(0).clip(lower=0)
    return pd.Series([0.0] * len(frame), index=frame.index)


def _compute_user_cumulative_unique_hotels(events: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(events), dtype=float)
    for _, idx in events.groupby("user_id", sort=False).groups.items():
        seen: set[str] = set()
        for pos in idx:
            seen.add(str(events.at[pos, "hotel_id"]))
            out[pos] = float(len(seen))
    return out


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def _build_weak_state_probs(events: pd.DataFrame) -> np.ndarray:
    lead = events["lead_time"].to_numpy(dtype=float)
    repeat_seen = (events["user_hotel_seen"] > 0).to_numpy(dtype=float)
    cold = (events["user_event_index"] <= 1).to_numpy(dtype=float)
    heavy_same_day = (events["user_day_queries"] >= 4).to_numpy(dtype=float)
    high_explore = (
        (events["user_cum_unique_hotels"] >= 4)
        & ((events["user_cum_unique_hotels"] / np.maximum(events["user_event_index"] + 1, 1)) >= 0.55)
    ).to_numpy(dtype=float)

    logits = np.zeros((len(events), K_STATES), dtype=float)

    # urgent_repeat
    logits[:, STATE_TO_IDX["urgent_repeat"]] = 2.0 * (lead <= 3) + 1.2 * repeat_seen + 0.4 * heavy_same_day
    # planned_repeat
    logits[:, STATE_TO_IDX["planned_repeat"]] = 1.8 * ((lead > 3) & (lead <= 30)) + 1.5 * repeat_seen
    # planned_explore
    logits[:, STATE_TO_IDX["planned_explore"]] = 1.4 * (1.0 - repeat_seen) + 1.4 * high_explore
    # price_probe
    logits[:, STATE_TO_IDX["price_probe"]] = 1.5 * heavy_same_day + 0.8 * (lead <= 15) + 0.6 * (1.0 - repeat_seen)
    # cold_start
    logits[:, STATE_TO_IDX["cold_start"]] = 2.2 * cold + 0.8 * (events["user_total_events"].to_numpy(dtype=float) <= 2)

    return _softmax_rows(logits)


def _estimate_transition_matrix(events: pd.DataFrame, q_weak: np.ndarray, alpha: float) -> np.ndarray:
    trans = np.full((K_STATES, K_STATES), float(max(alpha, 1e-6)), dtype=float)

    for _, idx in events.groupby("user_id", sort=False).groups.items():
        if len(idx) <= 1:
            continue
        seq_idx = list(idx)
        for p, c in zip(seq_idx[:-1], seq_idx[1:]):
            trans += np.outer(q_weak[p], q_weak[c])

    row_sums = np.clip(trans.sum(axis=1, keepdims=True), 1e-12, None)
    return trans / row_sums


def _markov_smooth(events: pd.DataFrame, q_weak: np.ndarray, eta: float, alpha: float) -> np.ndarray:
    eta = float(np.clip(eta, 0.0, 1.0))
    prior = np.clip(q_weak.mean(axis=0), 1e-9, None)
    prior = prior / prior.sum()

    A = _estimate_transition_matrix(events, q_weak, alpha=alpha)
    q_final = np.zeros_like(q_weak)

    for _, idx in events.groupby("user_id", sort=False).groups.items():
        seq_idx = list(idx)
        if not seq_idx:
            continue
        prev = prior
        for pos in seq_idx:
            pi = prev @ A
            fused = (np.clip(pi, 1e-9, None) ** eta) * (np.clip(q_weak[pos], 1e-9, None) ** (1.0 - eta))
            fused = fused / np.clip(fused.sum(), 1e-12, None)
            q_final[pos] = fused
            prev = fused

    return q_final


def build_midas_scores(
    requests_df: pd.DataFrame,
    p_reuse_df: pd.DataFrame,
    *,
    w_demand: float = 0.85,
    w_intent: float = 0.15,
    eta: float = 0.35,
    tau: float = 0.08,
    alpha: float = 1.0,
    eps: float = 1e-4,
) -> pd.DataFrame:
    """
    Build MIDAS score using latent behavior-state inference + Markov smoothing.

    Output columns:
    - cache_key
    - p_reuse
    - intent_score
    - markov_score
    - state_top
    - state_probs_json
    - state_confidence
    - state_entropy
    - midas_delta
    - midas_score
    - midas_version
    """
    if "cache_key" not in requests_df.columns:
        raise ValueError("requests_df must contain 'cache_key'.")
    if "rq_timestamp" not in requests_df.columns:
        raise ValueError("requests_df must contain 'rq_timestamp'.")
    if not {"cache_key", "p_reuse"}.issubset(set(p_reuse_df.columns)):
        raise ValueError("p_reuse_df must contain ['cache_key', 'p_reuse'].")

    params = MidasParams(
        w_demand=float(max(w_demand, 0.0)),
        w_intent=float(max(w_intent, 0.0)),
        eta=float(np.clip(eta, 0.0, 1.0)),
        tau=float(max(tau, 0.0)),
        alpha=float(max(alpha, 1e-6)),
        eps=float(np.clip(eps, 1e-9, 1e-2)),
    )

    # Normalize demand-side key score.
    p_reuse_base = (
        p_reuse_df[["cache_key", "p_reuse"]]
        .copy()
        .assign(cache_key=lambda x: x["cache_key"].astype(str))
        .assign(p_reuse=lambda x: pd.to_numeric(x["p_reuse"], errors="coerce"))
        .groupby("cache_key", as_index=False)["p_reuse"]
        .mean()
    )

    events = requests_df.copy()
    events["rq_timestamp"] = pd.to_datetime(events["rq_timestamp"], errors="coerce", utc=True)
    events = events.dropna(subset=["rq_timestamp", "cache_key"]).copy()
    events["cache_key"] = events["cache_key"].astype(str)

    events["user_id"] = _resolve_user_series(events)
    events["hotel_id"] = _resolve_hotel_series(events)
    events["lead_time"] = _resolve_lead_time(events)
    events["rq_date"] = events["rq_timestamp"].dt.date.astype(str)

    events = events.sort_values(["user_id", "rq_timestamp"], kind="stable").reset_index(drop=True)

    events["user_event_index"] = events.groupby("user_id", sort=False).cumcount()
    events["user_hotel_seen"] = events.groupby(["user_id", "hotel_id"], sort=False).cumcount()
    events["user_day_queries"] = events.groupby(["user_id", "rq_date"], sort=False)["cache_key"].transform("size")
    events["user_total_events"] = events.groupby("user_id", sort=False)["cache_key"].transform("size")
    events["user_cum_unique_hotels"] = _compute_user_cumulative_unique_hotels(events)

    q_weak = _build_weak_state_probs(events)
    q_final = _markov_smooth(events, q_weak, eta=params.eta, alpha=params.alpha)

    markov_score = (q_final @ STATE_REUSE_WEIGHTS).astype(float)
    state_idx = np.argmax(q_final, axis=1)
    state_top = [STATES[i] for i in state_idx]
    state_conf = q_final.max(axis=1)
    state_entropy = -(q_final * np.log(np.clip(q_final, 1e-12, None))).sum(axis=1)

    per_event = events[["cache_key"]].copy()
    per_event["markov_score"] = markov_score
    per_event["state_top"] = state_top
    per_event["state_confidence"] = state_conf
    per_event["state_entropy"] = state_entropy
    per_event["state_probs_json"] = [
        json.dumps({k: float(v) for k, v in zip(STATES, row)}, separators=(",", ":"))
        for row in q_final
    ]

    key_intent = (
        per_event.groupby("cache_key", as_index=False)
        .agg(
            markov_score=("markov_score", "mean"),
            state_confidence=("state_confidence", "mean"),
            state_entropy=("state_entropy", "mean"),
            state_top=("state_top", lambda s: s.value_counts().index[0] if len(s) else "cold_start"),
            state_probs_json=("state_probs_json", "last"),
        )
    )

    out = p_reuse_base.merge(key_intent, on="cache_key", how="left")
    out["p_reuse"] = pd.to_numeric(out["p_reuse"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    out["markov_score"] = pd.to_numeric(out["markov_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    out["state_confidence"] = pd.to_numeric(out["state_confidence"], errors="coerce").fillna(1.0 / K_STATES).clip(0.0, 1.0)
    out["state_entropy"] = pd.to_numeric(out["state_entropy"], errors="coerce").fillna(float(np.log(K_STATES)))
    out["state_top"] = out["state_top"].fillna("cold_start").astype(str)
    out["state_probs_json"] = out["state_probs_json"].fillna("{}")

    w_sum = params.w_demand + params.w_intent
    if w_sum <= 0:
        w_d, w_i = 0.85, 0.15
    else:
        w_d, w_i = params.w_demand / w_sum, params.w_intent / w_sum

    out["intent_score"] = (w_d * out["p_reuse"] + w_i * out["markov_score"]).clip(0.0, 1.0)

    centered = 1.6 * (out["intent_score"] - 0.5) + 0.7 * (out["state_confidence"] - (1.0 / K_STATES))
    out["midas_delta"] = params.tau * np.tanh(centered)
    out["midas_score"] = (out["p_reuse"] + out["midas_delta"]).clip(params.eps, 1.0 - params.eps)
    out["midas_version"] = "MIDAS_v2_latent_markov"

    return out[
        [
            "cache_key",
            "p_reuse",
            "intent_score",
            "markov_score",
            "state_top",
            "state_probs_json",
            "state_confidence",
            "state_entropy",
            "midas_delta",
            "midas_score",
            "midas_version",
        ]
    ]
