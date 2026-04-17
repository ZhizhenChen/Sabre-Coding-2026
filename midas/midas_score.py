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
    eta: float = 0.35
    tau: float = 0.08
    alpha: float = 1.0
    eps: float = 1e-4
    horizon_hours: float = 24.0
    train_val_ratio: float = 0.8
    train_epochs: int = 300
    train_lr: float = 0.05
    l2: float = 1e-3


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
    if "horizon" in frame.columns:
        return pd.to_numeric(frame["horizon"], errors="coerce").fillna(0).clip(lower=0)
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


def _markov_smooth(events: pd.DataFrame, q_weak: np.ndarray, eta: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
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

    return q_final, A


def _compute_reuse_target(events: pd.DataFrame, horizon_hours: float) -> np.ndarray:
    y = np.zeros(len(events), dtype=float)
    h = float(max(horizon_hours, 1e-6))
    for _, idx in events.groupby(["user_id", "hotel_id"], sort=False).groups.items():
        seq_idx = list(idx)
        if len(seq_idx) <= 1:
            continue
        t = events.loc[seq_idx, "rq_timestamp"].to_numpy(dtype="datetime64[ns]")
        next_t = np.roll(t, -1)
        dt_hours = (next_t - t).astype("timedelta64[s]").astype(float) / 3600.0
        hit = (dt_hours >= 0.0) & (dt_hours <= h)
        hit[-1] = False
        y[np.array(seq_idx, dtype=int)] = hit.astype(float)
    return y


def _compute_markov_features(events: pd.DataFrame, q_final: np.ndarray, A: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prior = np.clip(q_final.mean(axis=0), 1e-9, None)
    prior = prior / np.clip(prior.sum(), 1e-12, None)

    markov_score = (q_final @ STATE_REUSE_WEIGHTS).astype(float)
    m_t = np.zeros(len(events), dtype=float)

    for _, idx in events.groupby("user_id", sort=False).groups.items():
        seq_idx = list(idx)
        prev = prior
        for pos in seq_idx:
            cur = q_final[pos]
            val = float(prev @ A @ cur)
            m_t[pos] = max(val, eps)
            prev = cur

    l_t = np.log(np.clip(m_t, eps, None))
    h_t = -(q_final * np.log(np.clip(q_final, eps, None))).sum(axis=1)
    return markov_score, m_t, l_t, h_t


def _train_uplift_head(
    X: np.ndarray,
    p_demand: np.ndarray,
    y: np.ndarray,
    ts: pd.Series,
    tau: float,
    eps: float,
    train_val_ratio: float,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    n, d = X.shape
    if n == 0:
        return np.zeros(d, dtype=float), 0.0

    valid = np.isfinite(p_demand) & np.isfinite(y)
    if valid.sum() < max(50, d * 8):
        return np.zeros(d, dtype=float), 0.0

    X = X[valid]
    p_demand = np.clip(p_demand[valid], eps, 1.0 - eps)
    y = np.clip(y[valid], 0.0, 1.0)
    tsv = pd.to_datetime(ts[valid], utc=True, errors="coerce")

    if tsv.isna().all() or len(X) < 200:
        split = int(0.8 * len(X))
        tr_idx = np.arange(0, max(split, 1))
    else:
        cut = tsv.quantile(float(np.clip(train_val_ratio, 0.5, 0.95)))
        tr_idx = np.where(tsv <= cut)[0]
        if len(tr_idx) < 100:
            split = int(0.8 * len(X))
            tr_idx = np.arange(0, max(split, 1))

    Xt = X[tr_idx]
    pt = p_demand[tr_idx]
    yt = y[tr_idx]

    theta = np.zeros(d, dtype=float)
    bias = 0.0
    tau = float(max(tau, 1e-6))

    for _ in range(int(max(epochs, 1))):
        u = Xt @ theta + bias
        tnh = np.tanh(u)
        delta = tau * tnh
        p = np.clip(pt + delta, eps, 1.0 - eps)

        dL_dp = (p - yt) / np.clip(p * (1.0 - p), 1e-9, None)
        dp_du = tau * (1.0 - tnh * tnh)
        g = (dL_dp * dp_du) / max(len(Xt), 1)

        grad_theta = Xt.T @ g + 2.0 * float(max(l2, 0.0)) * theta
        grad_bias = float(g.sum())

        theta -= float(lr) * grad_theta
        bias -= float(lr) * grad_bias

    return theta, bias


def build_midas_scores(
    requests_df: pd.DataFrame,
    p_reuse_df: pd.DataFrame,
    *,
    w_demand: float = 0.85,
    w_markov: float = 0.15,
    eta: float = 0.35,
    tau: float = 0.08,
    alpha: float = 1.0,
    eps: float = 1e-4,
    horizon_hours: float = 24.0,
) -> pd.DataFrame:
    """Build MIDAS score using latent states + Markov dynamics + bounded uplift head.

    `w_demand` and `w_markov` are accepted for backward CLI compatibility only and are
    not used in the final score equation. Final score follows:
        p_final = clip(p_demand + tau * tanh(w^T phi + b), eps, 1-eps)
    where phi includes q_t (state probs), m_t, log(m_t), and state entropy.
    """
    _ = (w_demand, w_markov)

    if "cache_key" not in requests_df.columns:
        raise ValueError("requests_df must contain 'cache_key'.")
    if "rq_timestamp" not in requests_df.columns:
        raise ValueError("requests_df must contain 'rq_timestamp'.")
    if not {"cache_key", "p_reuse"}.issubset(set(p_reuse_df.columns)):
        raise ValueError("p_reuse_df must contain ['cache_key', 'p_reuse'].")

    params = MidasParams(
        eta=float(np.clip(eta, 0.0, 1.0)),
        tau=float(max(tau, 0.0)),
        alpha=float(max(alpha, 1e-6)),
        eps=float(np.clip(eps, 1e-9, 1e-2)),
        horizon_hours=float(max(horizon_hours, 1e-6)),
    )

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
    q_final, A = _markov_smooth(events, q_weak, eta=params.eta, alpha=params.alpha)

    markov_score, m_t, l_t, h_t = _compute_markov_features(events, q_final, A, eps=params.eps)

    p_event = events[["cache_key", "rq_timestamp"]].merge(p_reuse_base, on="cache_key", how="left")
    p_event["p_reuse"] = pd.to_numeric(p_event["p_reuse"], errors="coerce").fillna(0.5).clip(0.0, 1.0)

    y_event = _compute_reuse_target(events, horizon_hours=params.horizon_hours)

    X = np.column_stack([q_final, m_t, l_t, h_t])
    theta, bias = _train_uplift_head(
        X=X,
        p_demand=p_event["p_reuse"].to_numpy(dtype=float),
        y=y_event,
        ts=events["rq_timestamp"],
        tau=params.tau,
        eps=params.eps,
        train_val_ratio=params.train_val_ratio,
        epochs=params.train_epochs,
        lr=params.train_lr,
        l2=params.l2,
    )

    u = X @ theta + bias
    delta = params.tau * np.tanh(u)
    p_final = np.clip(p_event["p_reuse"].to_numpy(dtype=float) + delta, params.eps, 1.0 - params.eps)

    state_idx = np.argmax(q_final, axis=1)
    state_top = [STATES[i] for i in state_idx]
    state_conf = q_final.max(axis=1)
    state_entropy = h_t

    per_event = events[["cache_key"]].copy()
    per_event["p_reuse"] = p_event["p_reuse"].to_numpy(dtype=float)
    per_event["markov_score"] = markov_score
    per_event["m_t"] = m_t
    per_event["log_m_t"] = l_t
    per_event["state_top"] = state_top
    per_event["state_confidence"] = state_conf
    per_event["state_entropy"] = state_entropy
    per_event["intent_score"] = markov_score
    per_event["midas_delta"] = delta
    per_event["midas_score"] = p_final
    per_event["state_probs_json"] = [
        json.dumps({k: float(v) for k, v in zip(STATES, row)}, separators=(",", ":"))
        for row in q_final
    ]

    out = (
        per_event.groupby("cache_key", as_index=False)
        .agg(
            p_reuse=("p_reuse", "mean"),
            intent_score=("intent_score", "mean"),
            markov_score=("markov_score", "mean"),
            m_t=("m_t", "mean"),
            log_m_t=("log_m_t", "mean"),
            state_confidence=("state_confidence", "mean"),
            state_entropy=("state_entropy", "mean"),
            state_top=("state_top", lambda s: s.value_counts().index[0] if len(s) else "cold_start"),
            state_probs_json=("state_probs_json", "last"),
            midas_delta=("midas_delta", "mean"),
            midas_score=("midas_score", "mean"),
        )
        .sort_values("cache_key", kind="stable")
        .reset_index(drop=True)
    )

    out["midas_score"] = pd.to_numeric(out["midas_score"], errors="coerce").fillna(0.5).clip(params.eps, 1.0 - params.eps)
    out["p_reuse"] = pd.to_numeric(out["p_reuse"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    out["midas_delta"] = pd.to_numeric(out["midas_delta"], errors="coerce").fillna(0.0)
    out["midas_version"] = "MIDAS_v3_latent_markov_uplift"

    return out[
        [
            "cache_key",
            "p_reuse",
            "intent_score",
            "markov_score",
            "m_t",
            "log_m_t",
            "state_top",
            "state_probs_json",
            "state_confidence",
            "state_entropy",
            "midas_delta",
            "midas_score",
            "midas_version",
        ]
    ]
