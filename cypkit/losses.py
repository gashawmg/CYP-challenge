"""The dead-zone objective.

ST-RAE charges nothing for a prediction inside the fitted credible interval and the
distance to the nearest edge outside it. That is an L1 loss with a flat bottom. Squared
error is the wrong shape for it, and plain L1 is the wrong shape too — it charges for
being anywhere but the point estimate.

One residual definition covers both label kinds:

    r = 0                  if y_low <= p <= y_high
      = p - y_high         if p > y_high
      = p - y_low          if p < y_low

A measured compound gets its credible interval. A primary-screen non-hit gets
(-inf, floor], so it is free below the floor and charged above it — left-censored
regression, same code path.

Huberising near the edge keeps the Hessian positive so gradient boosting behaves;
`delta` -> 0 recovers the exact metric. Default 0.25 log units, well inside a typical
band width.
"""

from __future__ import annotations

import numpy as np


def dead_zone_residual(pred, y_low, y_high) -> np.ndarray:
    """Signed distance outside the band; zero inside it."""
    pred = np.asarray(pred, dtype=float)
    y_low = np.asarray(y_low, dtype=float)
    y_high = np.asarray(y_high, dtype=float)
    above = np.where(np.isfinite(y_high), np.clip(pred - y_high, 0.0, None), 0.0)
    below = np.where(np.isfinite(y_low), np.clip(y_low - pred, 0.0, None), 0.0)
    return above - below


def dead_zone_loss(pred, y_low, y_high, delta: float = 0.25, weight=None) -> np.ndarray:
    r = dead_zone_residual(pred, y_low, y_high)
    a = np.abs(r)
    loss = np.where(a <= delta, 0.5 * r**2 / max(delta, 1e-12), a - 0.5 * delta)
    if weight is not None:
        loss = loss * np.asarray(weight, dtype=float)
    return loss


def dead_zone_grad_hess(pred, y_low, y_high, delta: float = 0.25, weight=None):
    """Gradient and Hessian of the Huberised dead-zone loss."""
    r = dead_zone_residual(pred, y_low, y_high)
    a = np.abs(r)
    d = max(delta, 1e-12)
    grad = np.where(a <= d, r / d, np.sign(r))
    # The true Hessian is 1/delta inside the Huber band and 0 outside it, including
    # everywhere inside the dead zone. Handing a booster near-zero Hessians makes leaf
    # values (-sum g / sum h) explode, so we use a constant 1.0 — the same trick
    # LightGBM's own L1 objective uses. Step size is then governed by the learning
    # rate, and rows already inside the band contribute grad = 0, so they hold their
    # ground instead of being dragged.
    hess = np.ones_like(grad)
    if weight is not None:
        w = np.asarray(weight, dtype=float)
        grad = grad * w
        hess = hess * w
    return grad, hess


def make_lgb_objective(y_low, y_high, delta: float = 0.25, weight=None):
    """LightGBM custom objective closure.

    LightGBM passes (preds, dataset); labels are ignored because the target is the band,
    not a point. Pass the *same* row order used to build the Dataset.
    """
    y_low = np.asarray(y_low, dtype=float)
    y_high = np.asarray(y_high, dtype=float)
    w = None if weight is None else np.asarray(weight, dtype=float)

    def objective(preds, dataset):  # noqa: ARG001
        preds = np.asarray(preds, dtype=float).ravel()
        return dead_zone_grad_hess(preds, y_low, y_high, delta=delta, weight=w)

    return objective


def make_lgb_eval(y_low, y_high, delta: float = 0.25, weight=None, name="dead_zone"):
    """Matching eval metric so early stopping optimises the same thing."""
    y_low = np.asarray(y_low, dtype=float)
    y_high = np.asarray(y_high, dtype=float)
    w = None if weight is None else np.asarray(weight, dtype=float)

    def feval(preds, dataset):  # noqa: ARG001
        preds = np.asarray(preds, dtype=float).ravel()
        return name, float(np.mean(dead_zone_loss(preds, y_low, y_high, delta, w))), False

    return feval


def strae(y_true, pred, y_low, y_high) -> float:
    """Standalone ST-RAE, for quick checks that don't need the bootstrap machinery.

    The authoritative implementation is the challenge's own, wrapped in `scoring.py`.
    This exists so a training loop can print a number without a 1000-sample bootstrap.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    lo = np.asarray(y_low, dtype=float)
    hi = np.asarray(y_high, dtype=float)
    err = np.clip(pred - hi, 0, None) + np.clip(lo - pred, 0, None)
    m = np.mean(y_true)
    base = np.clip(m - hi, 0, None) + np.clip(lo - m, 0, None)
    denom = np.sum(base)
    return float(np.sum(err) / denom) if denom > 0 else float("nan")


def in_band_fraction(pred, y_low, y_high) -> float:
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(dead_zone_residual(pred, y_low, y_high) == 0.0))
