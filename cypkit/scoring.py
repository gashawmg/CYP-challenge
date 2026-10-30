"""Leaderboard-identical scoring, offline.

The challenge repo ships `evaluation/`, described in its own docstring as "ported from
the challenge backend, so it should match exactly what you see on the leaderboard". We
import that module rather than reimplementing it, so there is exactly one definition of
ST-RAE in this project and it is theirs.

The one thing the leaderboard has that we don't is ground truth for the test set. We
substitute out-of-fold predictions on the *training* set, which carries its own
`_conf_low` / `_conf_high` columns — the same credible-interval bounds ST-RAE needs.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


@contextmanager
def _challenge_on_path(repo: Path | None = None):
    repo = Path(repo or C.CHALLENGE_REPO)
    if not (repo / "evaluation" / "config.py").exists():
        raise FileNotFoundError(
            f"Could not find the challenge repo's evaluation/ package under {repo}.\n"
            "Set CYP_CHALLENGE_REPO to the directory that contains evaluation/."
        )
    added = str(repo) not in sys.path
    if added:
        sys.path.insert(0, str(repo))
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(str(repo))
            except ValueError:
                pass


QUIET = True  # the challenge module logs a paragraph per endpoint per call


def _import_challenge(repo: Path | None = None):
    with _challenge_on_path(repo):
        from evaluation import config as ev_config  # noqa: PLC0415
        from evaluation import evaluate_predictions as ev  # noqa: PLC0415
    if QUIET:
        try:
            from loguru import logger  # noqa: PLC0415

            logger.disable("evaluation")
        except ImportError:
            pass
    return ev, ev_config


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------


def build_ground_truth(
    train: pd.DataFrame,
    endpoints=C.REGRESSION_ENDPOINTS,
    subset: pd.Index | None = None,
) -> pd.DataFrame:
    """Reshape the training file into the ground-truth frame the scorer expects.

    Keeps the endpoint column plus its two credible-interval bounds. Rows with no
    measurement for an endpoint stay NaN — the scorer drops them per endpoint, exactly
    as it does for untested compounds in the real test set.
    """
    cols = [C.ID_COL]
    for ep in endpoints:
        lo, hi = ep + C.CONF_LOW_SUFFIX, ep + C.CONF_HIGH_SUFFIX
        if ep not in train.columns:
            raise KeyError(f"training frame has no column '{ep}'")
        cols.append(ep)
        for b in (lo, hi):
            if b in train.columns:
                cols.append(b)
    gt = train[cols].copy()
    if subset is not None:
        gt = gt.loc[gt.index.intersection(subset)]
    return gt.reset_index(drop=True)


def _has_bounds(gt: pd.DataFrame, endpoints) -> bool:
    return all(
        {ep + C.CONF_LOW_SUFFIX, ep + C.CONF_HIGH_SUFFIX} <= set(gt.columns)
        for ep in endpoints
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_regression(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    endpoints=C.REGRESSION_ENDPOINTS,
    n_bootstrap: int | None = None,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Score regression predictions and return the per-endpoint + macro summary.

    `predictions` and `ground_truth` both need `Molecule_Name` plus one column per
    endpoint. `ground_truth` additionally needs the `_conf_low` / `_conf_high` pair for
    ST-RAE; without them the metric silently degrades to plain RAE, so we refuse.

    Returns one row per endpoint (plus the macro row "MA"), with the mean and the 2.5 /
    97.5 percentiles across bootstrap samples for every metric.
    """
    endpoints = list(endpoints)
    if not _has_bounds(ground_truth, endpoints):
        raise ValueError(
            "ground_truth is missing _conf_low/_conf_high columns for at least one "
            "endpoint. Without them the challenge's ST-RAE collapses to plain RAE and "
            "the number you get back is not the leaderboard metric."
        )

    ev, ev_config = _import_challenge(repo)
    original = ev_config.BOOTSTRAP_SAMPLES
    if n_bootstrap is not None:
        ev_config.BOOTSTRAP_SAMPLES = int(n_bootstrap)
        ev.BOOTSTRAP_SAMPLES = int(n_bootstrap)
    try:
        with _challenge_on_path(repo):
            raw = ev.score_activity_predictions(
                predictions.copy(), ground_truth.copy(), endpoints
            )
            track = ev.add_macro_endpoint(raw, endpoints, ev_config.ACTIVITY_METRICS)
    finally:
        ev_config.BOOTSTRAP_SAMPLES = original
        ev.BOOTSTRAP_SAMPLES = original

    return _summarise(track, [name for name, _ in ev_config.ACTIVITY_METRICS])


def score_classification(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    endpoints=C.CLASSIFICATION_ENDPOINTS,
    n_bootstrap: int | None = None,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Same, for the TDI track. Predictions must already be booleans."""
    endpoints = list(endpoints)
    ev, ev_config = _import_challenge(repo)
    original = ev_config.BOOTSTRAP_SAMPLES
    if n_bootstrap is not None:
        ev_config.BOOTSTRAP_SAMPLES = int(n_bootstrap)
        ev.BOOTSTRAP_SAMPLES = int(n_bootstrap)
    try:
        with _challenge_on_path(repo):
            raw = ev.score_activity_predictions(
                predictions.copy(), ground_truth.copy(), endpoints
            )
            track = ev.add_macro_endpoint(raw, endpoints, ev_config.CLASSIFICATION_METRICS)
    finally:
        ev_config.BOOTSTRAP_SAMPLES = original
        ev.BOOTSTRAP_SAMPLES = original
    return _summarise(track, [name for name, _ in ev_config.CLASSIFICATION_METRICS])


def _summarise(track: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    metric_names = [m for m in metric_names if m in track.columns]
    g = track.groupby("Endpoint", sort=False)[metric_names]
    out = g.mean()
    for m in metric_names:
        q = track.groupby("Endpoint", sort=False)[m].quantile([0.025, 0.975]).unstack()
        out[m + "_lo"] = q[0.025]
        out[m + "_hi"] = q[0.975]
    ordered = [c for m in metric_names for c in (m, m + "_lo", m + "_hi")]
    out = out[ordered].astype(float).reset_index()
    return out


# ---------------------------------------------------------------------------
# Convenience for the OOF loop
# ---------------------------------------------------------------------------


def oof_report(
    oof: pd.DataFrame,
    train: pd.DataFrame,
    endpoints=C.REGRESSION_ENDPOINTS,
    n_bootstrap: int | None = None,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Score a frame of out-of-fold predictions against the training set's own bands.

    `oof` needs `Molecule_Name` and one column per endpoint. Compounds with no
    measurement for an endpoint are excluded from that endpoint by the scorer, so it is
    fine (and expected) for `oof` to carry a prediction for every compound.
    """
    gt = build_ground_truth(train, endpoints)
    keep = gt[C.ID_COL]
    preds = oof[oof[C.ID_COL].isin(keep)][[C.ID_COL, *endpoints]].copy()
    missing = set(keep) - set(preds[C.ID_COL])
    if missing:
        raise ValueError(
            f"{len(missing)} compound(s) in the ground truth have no prediction, "
            f"e.g. {sorted(missing)[:5]}"
        )
    return score_regression(preds, gt, endpoints, n_bootstrap=n_bootstrap, repo=repo)


def dense_proxy_report(
    oof: pd.DataFrame,
    label_matrix: pd.DataFrame,
    floor: float = C.PIC50_FLOOR,
    endpoints=C.REGRESSION_ENDPOINTS,
) -> pd.DataFrame:
    """Dead-zone error over the *full* (compound, isoform) matrix, censored rows included.

    Why this exists: `oof_report` is leaderboard-identical, but it can only score
    compounds that have a measured pIC50 — and a compound only has one because it hit
    the primary screen. So the honest local scorer inherits exactly the selection bias
    the censoring is meant to fix, and will mark down a model for learning that most
    compounds are inactive. The real test set was assayed densely and is full of them.

    This is a *proxy*, not the leaderboard metric: a censored row contributes the same
    one-sided error the metric would charge (free below the floor, charged above it),
    and enters the naive-baseline denominator at the floor. Report both numbers and
    never quote this one as ST-RAE.
    """
    out = []
    preds_by_id = {ep: dict(zip(oof[C.ID_COL], oof[ep])) for ep in endpoints}

    for ep in endpoints:
        rows = label_matrix[label_matrix["endpoint"] == ep]
        if rows.empty:
            continue
        p = rows[C.ID_COL].map(preds_by_id[ep]).to_numpy(dtype=float)
        keep = np.isfinite(p)
        rows, p = rows[keep], p[keep]
        lo = rows["y_low"].to_numpy(dtype=float)
        hi = rows["y_high"].to_numpy(dtype=float)
        y = rows["y"].to_numpy(dtype=float)
        y_ref = np.where(np.isfinite(y), y, floor)

        err = np.where(np.isfinite(hi), np.clip(p - hi, 0, None), 0.0) + np.where(
            np.isfinite(lo), np.clip(lo - p, 0, None), 0.0
        )
        m = float(np.mean(y_ref))
        base = np.where(np.isfinite(hi), np.clip(m - hi, 0, None), 0.0) + np.where(
            np.isfinite(lo), np.clip(lo - m, 0, None), 0.0
        )
        denom = float(np.sum(base))
        inside = float(np.mean(err == 0.0))
        out.append(
            dict(
                Endpoint=ep,
                dense_STRAE=float(np.sum(err) / denom) if denom > 0 else np.nan,
                in_band=inside,
                n=int(len(rows)),
                n_censored=int((rows["kind"] == "left").sum()),
            )
        )

    df = pd.DataFrame(out)
    if len(df):
        df.loc[len(df)] = dict(
            Endpoint="MA",
            dense_STRAE=df["dense_STRAE"].mean(),
            in_band=df["in_band"].mean(),
            n=int(df["n"].sum()),
            n_censored=int(df["n_censored"].sum()),
        )
    return df


SIMILARITY_BINS = (0.0, 0.40, 0.50, 0.60, 0.70, 1.01)


def _strae(y, p, lo, hi) -> float:
    """ST-RAE, same formula as the challenge's rae_soft_threshold_absolute_error.

    Used where the challenge scorer cannot be called directly — on resampled or binned
    subsets, where duplicated Molecule_Names would break its merge. `verify_strae()`
    checks this against the authoritative implementation.
    """
    err = np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)
    m = float(np.mean(y))
    base = np.clip(m - hi, 0, None) + np.clip(lo - m, 0, None)
    denom = float(np.sum(base))
    return float(np.sum(err) / denom) if denom > 0 else float("nan")


def verify_strae(oof, train, endpoint, repo=None, tol=1e-9) -> float:
    """Assert the local ST-RAE matches the challenge's, and return the difference."""
    gt = build_ground_truth(train, [endpoint])
    merged = gt.merge(oof[[C.ID_COL, endpoint]], on=C.ID_COL, suffixes=("_true", "_pred"))
    merged = merged.dropna(subset=[endpoint + "_true"])
    mine = _strae(
        merged[endpoint + "_true"].to_numpy(),
        merged[endpoint + "_pred"].to_numpy(),
        merged[endpoint + C.CONF_LOW_SUFFIX].to_numpy(),
        merged[endpoint + C.CONF_HIGH_SUFFIX].to_numpy(),
    )
    ev, _ = _import_challenge(repo)
    with _challenge_on_path(repo):
        from evaluation.custom_scoring_functions import (  # noqa: PLC0415
            rae_soft_threshold_absolute_error as theirs,
        )
    ref = theirs(
        merged[endpoint + "_true"].to_numpy(),
        merged[endpoint + "_pred"].to_numpy(),
        y_true_upper=merged[endpoint + C.CONF_HIGH_SUFFIX].to_numpy(),
        y_true_lower=merged[endpoint + C.CONF_LOW_SUFFIX].to_numpy(),
    )
    diff = abs(mine - ref)
    if diff > tol:
        raise AssertionError(f"local ST-RAE {mine} != challenge {ref} (diff {diff})")
    return diff


def similarity_report(
    oof: pd.DataFrame,
    train: pd.DataFrame,
    folds: pd.DataFrame,
    test_similarity: np.ndarray,
    endpoints=C.REGRESSION_ENDPOINTS,
    n_resamples: int = 200,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score OOF predictions against the similarity structure of the real test set.

    Two tables:

    `by_bin`   — ST-RAE within bands of "how close is this held-out compound to the rest
                 of the training set". Read the band the test set actually occupies.
    `matched`  — ST-RAE on held-out compounds resampled so their similarity histogram
                 matches the test set's, with a percentile interval over resamples.

    This exists because on this dataset the usual advice inverts: the test compounds are
    purchased analogues of screening hits and sit *closer* to the training set than
    training compounds sit to each other, so every structure-based split makes
    cross-validation harder than the real task rather than easier.
    """
    rng = np.random.default_rng(seed)
    sim_of = dict(zip(folds[C.ID_COL], folds["max_sim_to_other_folds"]))
    bins = np.asarray(SIMILARITY_BINS)
    test_hist, _ = np.histogram(test_similarity[np.isfinite(test_similarity)], bins=bins)
    test_frac = test_hist / max(test_hist.sum(), 1)

    by_bin, matched = [], []
    for ep in endpoints:
        gt = build_ground_truth(train, [ep])
        m = gt.merge(oof[[C.ID_COL, ep]], on=C.ID_COL, suffixes=("_true", "_pred"))
        m = m.dropna(subset=[ep + "_true"])
        m["sim"] = m[C.ID_COL].map(sim_of)
        m = m.dropna(subset=["sim"])
        y = m[ep + "_true"].to_numpy()
        p = m[ep + "_pred"].to_numpy()
        lo = m[ep + C.CONF_LOW_SUFFIX].to_numpy()
        hi = m[ep + C.CONF_HIGH_SUFFIX].to_numpy()
        sim = m["sim"].to_numpy()
        which = np.clip(np.digitize(sim, bins) - 1, 0, len(bins) - 2)

        for b in range(len(bins) - 1):
            idx = np.where(which == b)[0]
            by_bin.append(
                dict(
                    Endpoint=ep,
                    bin=f"{bins[b]:.2f}-{bins[b+1]:.2f}",
                    n=len(idx),
                    test_share=test_frac[b],
                    ST_RAE=_strae(y[idx], p[idx], lo[idx], hi[idx]) if len(idx) > 5 else np.nan,
                )
            )

        pools = [np.where(which == b)[0] for b in range(len(bins) - 1)]
        draws = []
        n_draw = len(m)
        for _ in range(n_resamples):
            take = []
            for b, frac in enumerate(test_frac):
                k = int(round(frac * n_draw))
                if k and len(pools[b]):
                    take.append(rng.choice(pools[b], size=k, replace=True))
            if not take:
                continue
            sel = np.concatenate(take)
            draws.append(_strae(y[sel], p[sel], lo[sel], hi[sel]))
        draws = np.asarray([d for d in draws if np.isfinite(d)])
        matched.append(
            dict(
                Endpoint=ep,
                matched_STRAE=float(draws.mean()) if len(draws) else np.nan,
                lo=float(np.percentile(draws, 2.5)) if len(draws) else np.nan,
                hi=float(np.percentile(draws, 97.5)) if len(draws) else np.nan,
                n_pool=len(m),
            )
        )

    mdf = pd.DataFrame(matched)
    if len(mdf):
        mdf.loc[len(mdf)] = dict(
            Endpoint="MA",
            matched_STRAE=mdf["matched_STRAE"].mean(),
            lo=mdf["lo"].mean(),
            hi=mdf["hi"].mean(),
            n_pool=int(mdf["n_pool"].sum()),
        )
    return pd.DataFrame(by_bin), mdf


def format_report(summary: pd.DataFrame, primary: str = "ST-RAE") -> str:
    """Compact console rendering, macro row last and flagged."""
    lines = []
    cols = [
        c
        for c in summary.columns
        if c != "Endpoint"
        and not c.endswith(("_lo", "_hi"))
        and pd.api.types.is_numeric_dtype(summary[c])
    ]
    width = max(len(str(e)) for e in summary["Endpoint"])
    header = "  ".join([f"{'Endpoint':<{width}}"] + [f"{c:>10}" for c in cols])
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in summary.iterrows():
        mark = " <-" if row["Endpoint"] == "MA" else ""
        lines.append(
            "  ".join(
                [f"{row['Endpoint']:<{width}}"]
                + [f"{row[c]:>10.4f}" if np.isfinite(row[c]) else f"{'nan':>10}" for c in cols]
            )
            + mark
        )
    if primary in summary.columns:
        ma = summary[summary["Endpoint"] == "MA"]
        if len(ma):
            r = ma.iloc[0]
            lines.append("")
            lines.append(
                f"MA-{primary} = {r[primary]:.4f}  "
                f"[95% CI {r[primary + '_lo']:.4f}, {r[primary + '_hi']:.4f}]"
            )
    return "\n".join(lines)


def regression_report(y, p, lo, hi) -> dict:
    """Every leaderboard metric for one endpoint, plus what placement could still buy.

    The leaderboard reports ST-RAE, MAE, R2, Spearman rho and Kendall tau. Reporting only
    the raw ST-RAE of a fold's predictions is misleading, because a submission is never
    sent raw: an affine map is fitted first, and it changes ST-RAE a great deal while
    leaving rho and tau untouched (CYP2D6 went 1.3092 -> 0.6602 on the leaderboard from
    placement alone, same model, same ordering).

    So this returns BOTH:

        strae_raw     what these predictions score as they stand
        strae_placed  the same ranking after the best a + b*p is fitted on this fold

    `strae_placed` is the number to watch while training, because it is the one a real
    submission approaches. It is also OPTIMISTIC: a and b are fitted on the very rows they
    are scored on, so it is an upper bound on what placement can achieve, not a forecast.
    The gap between the two columns is the placement work still on the table; the level of
    `strae_placed` is the model's own ceiling.
    """
    from scipy.stats import kendalltau, spearmanr  # noqa: PLC0415
    from scipy.optimize import minimize            # noqa: PLC0415

    y, p = np.asarray(y, float), np.asarray(p, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, lo, hi = y[ok], p[ok], lo[ok], hi[ok]
    lo = np.where(np.isfinite(lo), lo, y)
    hi = np.where(np.isfinite(hi), hi, y)
    if len(y) < 5:
        return {}

    rho = float(spearmanr(y, p).statistic)
    tau = float(kendalltau(y, p).statistic)
    ss = float(np.sum((y - y.mean()) ** 2))
    out = {
        "n": len(y),
        "rho": rho,
        "tau": tau,
        "r2": float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan"),
        "mae": float(np.mean(np.abs(y - p))),
        "strae_raw": _strae(y, p, lo, hi),
    }
    sp, sy = p.std(), y.std()
    b0 = (sy / sp) if sp > 1e-9 else 1.0
    r = minimize(lambda t: _strae(y, t[0] + t[1] * p, lo, hi),
                 x0=[y.mean() - b0 * p.mean(), b0], method="Nelder-Mead",
                 options=dict(maxiter=400, xatol=1e-3, fatol=1e-5))
    out["strae_placed"] = float(r.fun)
    out["a"], out["b"] = float(r.x[0]), float(r.x[1])
    return out


def format_report(rows: dict, title: str = "") -> str:
    """`rows` is {endpoint: regression_report(...)}. Returns a printable block."""
    if not rows:
        return "  (no scored labels in this fold)"
    keys = [k for k in rows if rows[k]]
    w = ["endpoint", "n", "rho", "tau", "R2", "MAE", "ST-RAE", "placed"]
    out = []
    if title:
        out.append(title)
    out.append(f"  {w[0]:9s} {w[1]:>5s} {w[2]:>7s} {w[3]:>7s} {w[4]:>7s} "
               f"{w[5]:>7s} {w[6]:>8s} {w[7]:>8s}")
    for k in keys:
        d = rows[k]
        out.append(f"  {k:9s} {d['n']:5d} {d['rho']:7.4f} {d['tau']:7.4f} "
                   f"{d['r2']:7.3f} {d['mae']:7.3f} {d['strae_raw']:8.3f} "
                   f"{d['strae_placed']:8.4f}")
    if len(keys) > 1:
        m = lambda f: np.mean([rows[k][f] for k in keys])  # noqa: E731
        out.append(f"  {'MACRO':9s} {'':5s} {m('rho'):7.4f} {m('tau'):7.4f} "
                   f"{m('r2'):7.3f} {m('mae'):7.3f} {m('strae_raw'):8.3f} "
                   f"{m('strae_placed'):8.4f}")
    out.append("  'placed' = ST-RAE after fitting a+b*p ON THIS FOLD -- an upper bound on "
               "what placement can\n  reach, and the column to watch. rho and tau are "
               "untouched by placement.")
    return "\n".join(out)
