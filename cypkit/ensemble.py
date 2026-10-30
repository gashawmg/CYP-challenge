"""Track A: a genuinely diverse classical-ML ensemble.

The plateau diagnosis was that every model we had ran on *one* fixed 2,482-column
feature vector and differed only in the loss. Blending near-duplicates buys nothing,
which is why the phase-2 blends all recalibrated to within noise of the single model.

So diversity here is structural, not cosmetic. Each learner gets its own
representation as well as its own inductive bias:

    lgb_dz    LightGBM, dead-zone objective     rdkit + cyp + ecfp + maccs
    lgb_l1    LightGBM, MAE                     rdkit + cyp + ap
    xgb_dz    XGBoost,  dead-zone objective     rdkit + cyp            (dense only)
    xgb_l1    XGBoost,  MAE                     rdkit + cyp + fcfp
    cat_l1    CatBoost, MAE (ordered boosting)  rdkit + cyp + maccs
    svr_tan   SVR on a Tanimoto kernel          ecfp                   (kernel method)
    knn_tan   Tanimoto-weighted kNN             ecfp                   (pure similarity)
    et_l1     ExtraTrees                        rdkit + cyp + maccs

Three of those (SVR, kNN, ExtraTrees) cannot express the dead-zone loss at all, so
they train on floor-imputed labels. That is fine and deliberate: they are in the pool
to disagree with the boosters, and the blend decides how much to listen.

Every learner sees the censored rows. The primary screen is the only dense label
matrix in this dataset and dropping it is what made our predictions unbiased on
actives and 0.9 log units too high on inactives.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C
from . import losses

# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

LEARNERS: dict[str, dict] = {
    "lgb_dz": dict(
        kind="lgb", loss="deadzone", features="rdkit+cyp+prior+ecfp+maccs",
        params=dict(learning_rate=0.20, num_leaves=63, min_data_in_leaf=10,
                    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                    n_rounds=600, verbose=-1, seed=42),
    ),
    "lgb_l1": dict(
        kind="lgb", loss="l1", features="rdkit+cyp+prior+ap",
        params=dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
                    feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
                    n_rounds=400, verbose=-1, seed=7),
    ),
    "xgb_dz": dict(
        kind="xgb", loss="deadzone", features="rdkit+cyp+prior",
        params=dict(eta=0.20, max_depth=7, subsample=0.8, colsample_bytree=0.6,
                    min_child_weight=5, reg_lambda=1.0, n_rounds=500,
                    tree_method="hist", nthread=0),
    ),
    "xgb_l1": dict(
        kind="xgb", loss="l1", features="rdkit+cyp+prior+fcfp",
        params=dict(eta=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.4,
                    min_child_weight=10, reg_lambda=2.0, n_rounds=600,
                    tree_method="hist", nthread=0),
    ),
    "cat_l1": dict(
        kind="cat", loss="l1", features="rdkit+cyp+prior+maccs",
        params=dict(iterations=800, learning_rate=0.06, depth=8,
                    l2_leaf_reg=3.0, random_seed=13, verbose=0,
                    loss_function="MAE", allow_writing_files=False),
    ),
    "svr_tan": dict(
        kind="svr", loss="l1", features="ecfp",
        params=dict(C=3.0, epsilon=0.15, max_train=6000),
    ),
    "knn_tan": dict(
        kind="knn", loss="l1", features="ecfp",
        params=dict(k=15, power=3.0),
    ),
    "et_l1": dict(
        kind="et", loss="l1", features="rdkit+cyp+prior+maccs",
        params=dict(n_estimators=500, min_samples_leaf=2, max_features=0.3,
                    n_jobs=-1, random_state=21),
    ),
}


def blocks_needed(names) -> list[str]:
    """Union of feature blocks the chosen learners need, in a stable order."""
    order = ["rdkit", "cyp", "prior", "tdialert", "ecfp", "ecfpc", "fcfp", "maccs", "ap", "3d"]
    want = set()
    for n in names:
        want.update(LEARNERS[n]["features"].split("+"))
    return [b for b in order if b in want]


class FeatureBank:
    """One featurisation pass; every learner takes a column slice of it."""

    def __init__(self, ids: list[str], X: np.ndarray, spans: dict[str, tuple[int, int]]):
        self.ids = ids
        self.X = X
        self.spans = spans
        self.index_of = {m: i for i, m in enumerate(ids)}
        self._cache: dict[str, np.ndarray] = {}

    def view(self, feature_spec: str) -> np.ndarray:
        if feature_spec in self._cache:
            return self._cache[feature_spec]
        cols = []
        for b in feature_spec.split("+"):
            if b not in self.spans:
                raise KeyError(f"feature block '{b}' was not built into the bank")
            lo, hi = self.spans[b]
            cols.append(np.arange(lo, hi))
        idx = np.concatenate(cols)
        V = np.ascontiguousarray(self.X[:, idx])
        self._cache[feature_spec] = V
        return V


# ---------------------------------------------------------------------------
# Tanimoto machinery
# ---------------------------------------------------------------------------


def tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto between two binary fingerprint matrices."""
    A = (A > 0).astype(np.float32)
    B = (B > 0).astype(np.float32)
    inter = A @ B.T
    a = A.sum(1)[:, None]
    b = B.sum(1)[None, :]
    denom = a + b - inter
    return np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)


# ---------------------------------------------------------------------------
# Label preparation
# ---------------------------------------------------------------------------


# ST-RAE charges nothing inside a compound's fitted credible interval, so a wide
# interval is a wide dead zone and a narrow one is nearly a point target. Measured on
# our submitted blend: the NARROWEST quartile of intervals is 25% of the rows and
# 38-60% of the total ST-RAE cost, with only 5-11% of those rows landing inside.
# Training every row at equal weight optimises squared error against a metric that
# does not charge like squared error. Weighting by 1/width moved macro OOF ST-RAE
# 0.4576 -> 0.4489 (power 1.0) with the narrow-quartile cost falling as intended.
# Power 0 restores the old behaviour; the cap stops a handful of very tight intervals
# dominating the fit.
INTERVAL_WEIGHT_POWER = float(os.environ.get("CYP_INTERVAL_WEIGHT_POWER", 1.0))
INTERVAL_WEIGHT_CAP = 4.0


def interval_weights(rows: pd.DataFrame, power: float | None = None) -> np.ndarray:
    """Multiplier per row from its credible-interval width. 1.0 where there is none."""
    power = INTERVAL_WEIGHT_POWER if power is None else power
    n = len(rows)
    if power == 0 or "y_low" not in rows or "y_high" not in rows:
        return np.ones(n)
    lo = rows["y_low"].to_numpy(dtype=float)
    hi = rows["y_high"].to_numpy(dtype=float)
    width = hi - lo
    ok = np.isfinite(width) & (width > 0)
    if ok.sum() < 20:
        return np.ones(n)
    med = float(np.median(width[ok]))
    out = np.ones(n)
    out[ok] = np.clip((med / width[ok]) ** power,
                      1.0 / INTERVAL_WEIGHT_CAP, INTERVAL_WEIGHT_CAP)
    return out


def _point_labels(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Point estimate with censored rows imputed at the floor, plus row weights."""
    y = rows["y"].to_numpy(dtype=float)
    hi = rows["y_high"].to_numpy(dtype=float)
    y = np.where(np.isfinite(y), y, np.where(np.isfinite(hi), hi, C.PIC50_FLOOR))
    w = rows["weight"].to_numpy(dtype=float) if "weight" in rows else np.ones(len(rows))
    return y, w * interval_weights(rows)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_predict(name: str, Xtr: np.ndarray, rows: pd.DataFrame, Xte: np.ndarray,
                delta: float = 0.25) -> np.ndarray | None:
    """Train one learner on one (isoform, fold) split and predict Xte."""
    spec = LEARNERS[name]
    if len(rows) < 20:
        return None
    kind, loss, p = spec["kind"], spec["loss"], dict(spec["params"])

    measured = rows["y"].to_numpy(dtype=float)
    measured = measured[np.isfinite(measured)]
    init = float(np.median(measured)) if len(measured) else C.PIC50_FLOOR

    if kind == "lgb":
        return _fit_lgb(loss, Xtr, rows, Xte, p, delta, init)
    if kind == "xgb":
        return _fit_xgb(loss, Xtr, rows, Xte, p, delta, init)
    if kind == "cat":
        return _fit_cat(Xtr, rows, Xte, p)
    if kind == "et":
        return _fit_et(Xtr, rows, Xte, p)
    if kind == "svr":
        return _fit_svr(Xtr, rows, Xte, p)
    if kind == "knn":
        return _fit_knn(Xtr, rows, Xte, p)
    raise ValueError(f"unknown learner kind {kind!r}")


def _fit_lgb(loss, Xtr, rows, Xte, p, delta, init):
    import lightgbm as lgb  # noqa: PLC0415

    n_rounds = p.pop("n_rounds")
    if loss == "l1":
        y, w = _point_labels(rows)
        p["objective"] = "regression_l1"
        m = lgb.train(p, lgb.Dataset(Xtr, label=y, weight=w), num_boost_round=n_rounds)
        return m.predict(Xte)

    lo = rows["y_low"].to_numpy(dtype=float)
    hi = rows["y_high"].to_numpy(dtype=float)
    w = rows["weight"].to_numpy(dtype=float)
    p["objective"] = losses.make_lgb_objective(lo, hi, delta=delta, weight=w)
    ds = lgb.Dataset(Xtr, label=np.zeros(len(lo)), init_score=np.full(len(lo), init))
    m = lgb.train(p, ds, num_boost_round=n_rounds)
    return m.predict(Xte) + init


def _fit_xgb(loss, Xtr, rows, Xte, p, delta, init):
    import xgboost as xgb  # noqa: PLC0415

    n_rounds = p.pop("n_rounds")
    dte = xgb.DMatrix(Xte)

    if loss == "l1":
        y, w = _point_labels(rows)
        dtr = xgb.DMatrix(Xtr, label=y, weight=w)
        p = {**p, "objective": "reg:absoluteerror", "base_score": init}
        m = xgb.train(p, dtr, num_boost_round=n_rounds)
        return m.predict(dte)

    lo = rows["y_low"].to_numpy(dtype=float)
    hi = rows["y_high"].to_numpy(dtype=float)
    w = rows["weight"].to_numpy(dtype=float)
    dtr = xgb.DMatrix(Xtr, label=np.zeros(len(lo)))
    dtr.set_base_margin(np.full(len(lo), init))
    dte.set_base_margin(np.full(Xte.shape[0], init))

    def obj(pred, _dm):
        return losses.dead_zone_grad_hess(np.asarray(pred, float).ravel(), lo, hi,
                                          delta=delta, weight=w)

    m = xgb.train({**p, "disable_default_eval_metric": 1}, dtr,
                  num_boost_round=n_rounds, obj=obj)
    return m.predict(dte)


def _fit_cat(Xtr, rows, Xte, p):
    from catboost import CatBoostRegressor  # noqa: PLC0415

    y, w = _point_labels(rows)
    m = CatBoostRegressor(**p)
    m.fit(Xtr, y, sample_weight=w)
    return m.predict(Xte)


def _fit_et(Xtr, rows, Xte, p):
    from sklearn.ensemble import ExtraTreesRegressor  # noqa: PLC0415

    y, w = _point_labels(rows)
    m = ExtraTreesRegressor(**p)
    m.fit(Xtr, y, sample_weight=w)
    return m.predict(Xte)


def _fit_svr(Xtr, rows, Xte, p):
    from sklearn.svm import SVR  # noqa: PLC0415

    y, w = _point_labels(rows)
    max_train = int(p.pop("max_train", 6000))
    if len(y) > max_train:
        # Keep every measured row; subsample the censored ones, which are cheap and
        # numerous. A quadratic-memory kernel method cannot take all 15k rows.
        meas = np.where(np.isfinite(rows["y"].to_numpy(dtype=float)))[0]
        cens = np.where(~np.isfinite(rows["y"].to_numpy(dtype=float)))[0]
        room = max(max_train - len(meas), 0)
        rng = np.random.default_rng(0)
        keep = np.concatenate([meas, rng.choice(cens, size=min(room, len(cens)), replace=False)])
        keep.sort()
        Xtr, y, w = Xtr[keep], y[keep], w[keep]
    K = tanimoto(Xtr, Xtr).astype(np.float64)
    m = SVR(kernel="precomputed", **p)
    m.fit(K, y, sample_weight=w)
    return m.predict(tanimoto(Xte, Xtr).astype(np.float64))


def _fit_knn(Xtr, rows, Xte, p):
    y, _ = _point_labels(rows)
    k = int(p.get("k", 15))
    power = float(p.get("power", 3.0))
    out = np.empty(Xte.shape[0], dtype=float)
    step = 512
    for s in range(0, Xte.shape[0], step):
        S = tanimoto(Xte[s:s + step], Xtr)
        k_eff = min(k, S.shape[1])
        idx = np.argpartition(-S, k_eff - 1, axis=1)[:, :k_eff]
        sims = np.take_along_axis(S, idx, axis=1) ** power
        vals = y[idx]
        wsum = sims.sum(1)
        out[s:s + step] = np.where(wsum > 0, (sims * vals).sum(1) / np.maximum(wsum, 1e-12),
                                   float(np.median(y)))
    return out


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------


def blend_weights(P: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  weight: np.ndarray | None = None, delta: float = 0.05,
                  n_restarts: int = 4) -> np.ndarray:
    """Non-negative simplex weights minimising the dead-zone loss on OOF predictions.

    `P` is (n_rows, n_learners). Optimised on the FULL label matrix — censored rows
    included — because that is the distribution the blind test set was drawn from,
    and the leaderboard-identical scorer cannot see it.
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    n = P.shape[1]

    def f(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        if s <= 0:
            return 1e9
        p = P @ (w / s)
        return float(np.mean(losses.dead_zone_loss(p, lo, hi, delta=delta, weight=weight)))

    best = None
    rng = np.random.default_rng(0)
    starts = [np.ones(n) / n] + [rng.dirichlet(np.ones(n)) for _ in range(n_restarts - 1)]
    for w0 in starts:
        r = minimize(f, w0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    w = np.clip(best.x, 0, None)
    return w / max(w.sum(), 1e-12)
