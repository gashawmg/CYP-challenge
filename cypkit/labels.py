"""The censored label matrix.

Dose-response curves were only run on compounds that hit the primary screen, so every
pIC50 in the training file is conditioned on already being active. The test set has no
such filter. A model fit on curve rows alone has never seen an inactive and will
over-predict potency across most of the test set.

The primary screen covers every compound against all four isoforms, so a non-hit is not
missing data — it is an observation that the true pIC50 lies below the assay floor. This
module turns both kinds of row into one long-format table:

    kind = "interval"  -> y_low/y_high are the fitted credible interval  (measured)
    kind = "left"      -> y_high is the floor, y_low is -inf             (non-hit)

Both are consumed by the same objective in `losses.py`: no penalty inside [y_low, y_high],
linear outside it. For a left-censored row that means "free to predict anything below the
floor, charged for predicting above it", which is exactly what a non-hit tells you.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def hits_from_screen(
    single_conc: pd.DataFrame,
    log2fc_max: float = C.HIT_LOG2FC_MAX,
    fdr_max: float = C.HIT_FDR_MAX,
) -> pd.DataFrame:
    """Add the tutorial's hit flag to the single-concentration frame."""
    required = ["enzyme", "log2fc_estimate", "log2fc_fdr", C.ID_COL]
    missing = [c for c in required if c not in single_conc.columns]
    if missing:
        raise KeyError(f"single-concentration frame is missing {missing}")
    out = single_conc.copy()
    out["is_hit"] = (out["log2fc_estimate"] < log2fc_max) & (out["log2fc_fdr"] < fdr_max)
    return out


class FoldedIsotonic:
    """A log2fc -> pIC50 map that knows which fold it may not have learned from.

    WHY THIS EXISTS. The previous `pseudo_from_screen` fitted ONE isotonic map on every
    compound that has both a screen reading and a fitted curve -- including the compounds
    in the fold being predicted. The pseudo-labels handed to the training folds were
    therefore shaped by the held-out fold's own curves, which leaks into every
    out-of-fold estimate built on them. scripts/screen_pseudo.py always did this per fold;
    the function the training scripts actually call did not (found 2026-09-26, Section 33).

    Measured size of the leak, refitting leave-one-fold-out and comparing the labels
    produced: mean shift 0.042 / 0.027 / 0.098 / 0.022 log units on 1A2 / 2C9 / 2D6 / 3A4,
    against a map residual sd of 0.42 / 0.41 / 0.63 / 0.41. Small -- 5% to 16% of the noise
    the pseudo-labels already carry -- but free to remove and systematically optimistic.

    `predict(x, fold)` uses the map fitted WITHOUT that fold. `fold=None` falls back to the
    all-data map, which is correct at inference time (the blind compounds contribute no
    curves, so nothing of theirs can be in any map) and wrong during cross-validation.
    """

    def __init__(self, full, per_fold=None):
        self.full = full
        self.per_fold = per_fold or {}

    def predict(self, x, fold=None):
        m = self.per_fold.get(fold, self.full) if fold is not None else self.full
        return m.predict(x)


def pseudo_from_screen(
    train_inhibition: pd.DataFrame,
    single_conc: pd.DataFrame,
    band: float = 1.0,
    verbose: bool = True,
    fold_of: dict | None = None,
) -> dict[str, tuple]:
    """Fit a monotone log2fc -> pIC50 map per isoform on the compounds that have both.

    The screen's `log2fc_estimate` is not a weak proxy for potency -- on the overlap it
    tracks pIC50 at Spearman -0.83 to -0.94, which on CYP2D6 is a stronger relationship
    than our own best model achieves against the blind set. What it lacks is a scale, and
    a single-concentration readout saturates at the potent end, so the map has to be
    monotone but free in shape: isotonic regression, no functional form assumed.

    Returns {isoform: (map, residual_sd)}. The residual sd is what makes these labels
    honest downstream -- a pseudo-label enters as an INTERVAL of +/- band*sd, not a
    point, so the same interval-aware objective that handles fitted curves and censored
    non-hits handles these too, and the ST-RAE interval weighting discounts them
    automatically instead of by a weight someone chose by hand.
    """
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

    g = (single_conc.groupby([C.ID_COL, "enzyme"])["log2fc_estimate"].mean()
         .reset_index().rename(columns={"log2fc_estimate": "lfc"}))
    out = {}
    for cyp in C.ISOFORMS:
        ep = f"{cyp}_pIC50_direct_inhibition"
        if ep not in train_inhibition.columns:
            continue
        real = train_inhibition.loc[train_inhibition[ep].notna(), [C.ID_COL, ep]]
        m = real.merge(g[g["enzyme"] == cyp], on=C.ID_COL)
        if len(m) < 100:
            continue
        lfc_a, y_a = m["lfc"].to_numpy(float), m[ep].to_numpy(float)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(lfc_a, y_a)
        sd = float(np.std(y_a - iso.predict(lfc_a)))
        per_fold = {}
        if fold_of:
            f = m[C.ID_COL].map(fold_of).to_numpy()
            for fv in pd.unique(f[pd.notna(f)]):
                keep = f != fv
                if keep.sum() < 100:
                    continue          # too thin to refit; predict() falls back to `full`
                it = IsotonicRegression(increasing=False, out_of_bounds="clip")
                it.fit(lfc_a[keep], y_a[keep])
                per_fold[fv] = it
        out[cyp] = (FoldedIsotonic(iso, per_fold), sd)
        if verbose:
            from scipy.stats import spearmanr  # noqa: PLC0415
            print(f"    {cyp}: map fitted on {len(m)} overlap compounds, "
                  f"screen rho {spearmanr(m['lfc'], m[ep])[0]:+.3f}, residual sd {sd:.3f}"
                  + (f", + {len(per_fold)} leave-one-fold-out maps" if per_fold
                     else "  [NO FOLDS GIVEN -- leaks into out-of-fold estimates]"))
    return out


def build_label_matrix(
    train_inhibition: pd.DataFrame,
    single_conc: pd.DataFrame,
    floor: float = C.PIC50_FLOOR,
    include_censored: bool = True,
    censored_weight: float = 1.0,
    clip_interval_at_floor: bool = True,
    keep_subfloor: tuple = ("CYP3A4",),
    pseudo: bool = False,
    pseudo_band: float = 1.0,
    pseudo_weight: float = 0.5,
    pseudo_nonhit_interval: bool = False,
    pseudo_fold_of: dict | None = None,
) -> pd.DataFrame:
    """Long-format targets, one row per (compound, isoform).

    Args:
        floor: pIC50 below which the assay cannot quantify. Non-hits are left-censored
            here, and measured values at or below it are treated as censored too rather
            than as points, since they are not really measurements.
        include_censored: set False to reproduce the DRC-only training set.
        censored_weight: relative weight for screen-derived rows. There are many more of
            them than curves; below 1.0 stops them dominating the gradient.
        clip_interval_at_floor: a fitted curve whose credible interval extends below the
            floor is only telling you "weak" — widen the band to -inf below the floor
            instead of pretending the lower bound is meaningful.

    Returns columns:
        Molecule_Name, SMILES, isoform, endpoint, y, y_low, y_high, kind, weight, source
    """
    screen = hits_from_screen(single_conc)
    maps = pseudo_from_screen(train_inhibition, single_conc, band=pseudo_band,
                              fold_of=pseudo_fold_of) if pseudo else {}
    lfc_lookup = {}
    if pseudo:
        agg = (single_conc.groupby([C.ID_COL, "enzyme"])["log2fc_estimate"]
               .mean().reset_index())
        lfc_lookup = {(r.Molecule_Name, r.enzyme): float(r.log2fc_estimate)
                      for r in agg.itertuples(index=False)}
    hit_lookup = {
        (r.Molecule_Name, r.enzyme): bool(r.is_hit)
        for r in screen.itertuples(index=False)
    }
    screened = set(screen[C.ID_COL])

    smiles_lookup = {}
    for frame in (train_inhibition, screen):
        if C.SMILES_COL in frame.columns:
            smiles_lookup.update(dict(zip(frame[C.ID_COL], frame[C.SMILES_COL])))

    rows = []
    for cyp in C.ISOFORMS:
        ep = f"{cyp}_pIC50_direct_inhibition"
        lo_col, hi_col = ep + C.CONF_LOW_SUFFIX, ep + C.CONF_HIGH_SUFFIX
        has_bounds = {lo_col, hi_col} <= set(train_inhibition.columns)

        measured_ids = set()
        if ep in train_inhibition.columns:
            sub = train_inhibition[train_inhibition[ep].notna()]
            for r in sub.itertuples(index=False):
                y = float(getattr(r, ep))
                lo = float(getattr(r, lo_col)) if has_bounds else y
                hi = float(getattr(r, hi_col)) if has_bounds else y
                if not np.isfinite(lo):
                    lo = y
                if not np.isfinite(hi):
                    hi = y
                lo, hi = min(lo, hi), max(lo, hi)
                kind = "interval"
                # Keeping a sub-floor curve only pays where there are ENOUGH of them to
                # learn the region from. Measured out-of-fold on ChemProp v8, Spearman on
                # the recovered rows themselves:
                #     CYP3A4  944 recovered (40.4%)  rho 0.542   MAE 0.972 -> 0.738
                #     CYP2C9  259 recovered (20.2%)  rho 0.393
                #     CYP1A2  232 recovered (16.4%)  rho 0.158
                #     CYP2D6  129 recovered ( 8.6%)  rho -0.041  <- pure noise
                # On CYP3A4 the fix is clearly right. On CYP2D6, 129 wide-band rows are
                # too few to teach the sub-floor ordering and the model cannot rank them
                # at all, so they inject noise instead. `keep_subfloor` names the
                # isoforms where the evidence supports keeping them.
                keep_here = isinstance(keep_subfloor, (list, tuple, set)) \
                    and cyp in keep_subfloor
                if clip_interval_at_floor and (y <= floor or hi <= floor) \
                        and not (keep_here and has_bounds
                                 and np.isfinite(lo) and np.isfinite(hi)):
                    # A curve BELOW the floor is still a curve. 1,564 of 6,525 fitted
                    # values sit at or under 4.0 (CYP3A4 alone: 944, 40.4%), and every
                    # one of them carries finite _conf_low/_conf_high -- CYP3A4's average
                    # [1.99, 3.79]. `scoring.build_ground_truth` charges ST-RAE against
                    # exactly those bounds, so widening them to -inf told the model
                    # "anything below 4.0 is free" for a fifth of the training set while
                    # the metric was charging the moment the prediction left the band.
                    # That is a training/metric mismatch, not a modelling choice.
                    #
                    # PIC50_FLOOR = 4.0 came from the TDI label rule, where it is the
                    # right constant; it was never validated as a quantification limit
                    # for the direct-inhibition curves, and the data says it is not one.
                    #
                    # So censoring is now reserved for rows that genuinely have no
                    # curve -- screen non-hits, handled further down. A fitted value
                    # with finite bounds keeps them wherever it sits.
                    lo, kind = -np.inf, "left"
                    hi = max(hi, floor)
                mid = getattr(r, C.ID_COL)
                measured_ids.add(mid)
                rows.append(
                    dict(
                        Molecule_Name=mid,
                        SMILES=smiles_lookup.get(mid),
                        isoform=cyp,
                        endpoint=ep,
                        y=y,
                        y_low=lo,
                        y_high=hi,
                        kind=kind,
                        weight=1.0,
                        source="drc",
                    )
                )

        if not include_censored:
            continue

        for mid in screened:
            if mid in measured_ids:
                continue
            hit = hit_lookup.get((mid, cyp))
            if hit is None:
                continue          # not screened against this isoform
            lfc = lfc_lookup.get((mid, cyp))
            if cyp in maps and lfc is not None and np.isfinite(lfc):
                # The screen said something quantitative about this compound; use it.
                # A non-hit becomes an interval centred where the map puts it rather
                # than a flat "< floor", and a HIT WITHOUT A CURVE -- previously thrown
                # away entirely, and the most active compounds among the unlabelled --
                # finally enters training at all.
                mp, sd = maps[cyp]
                # the fold this compound sits in -- its own curves must not have shaped
                # the map that labels it
                est = float(mp.predict([lfc],
                                       (pseudo_fold_of or {}).get(mid))[0])
                lo, hi = est - pseudo_band * sd, est + pseudo_band * sd
                if not hit and not pseudo_nonhit_interval:
                    # Default: a non-hit stays a BOUND. The map's estimate for it sits
                    # below the assay floor, where no number is real, and the blind set
                    # is more active than training -- inventing sub-floor precision is
                    # not where the metric pays. `pseudo_nonhit_interval=True` is the
                    # aggressive variant the LightGBM attribution test used, where every
                    # screen-only compound became a point target.
                    lo, hi = -np.inf, max(hi, floor)
                rows.append(
                    dict(
                        Molecule_Name=mid,
                        SMILES=smiles_lookup.get(mid),
                        isoform=cyp,
                        endpoint=ep,
                        y=est,
                        y_low=lo,
                        y_high=hi,
                        kind="interval" if (hit or pseudo_nonhit_interval) else "left",
                        weight=pseudo_weight,
                        source="pseudo",
                    )
                )
                continue
            if hit:
                # It *was* a hit but has no curve — without a map we know too little
                # to censor safely.
                continue
            rows.append(
                dict(
                    Molecule_Name=mid,
                    SMILES=smiles_lookup.get(mid),
                    isoform=cyp,
                    endpoint=ep,
                    y=np.nan,
                    y_low=-np.inf,
                    y_high=floor,
                    kind="left",
                    weight=censored_weight,
                    source="screen",
                )
            )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["isoform", "source", C.ID_COL]).reset_index(drop=True)
    return out


def summarise(labels: pd.DataFrame) -> pd.DataFrame:
    """Rows per isoform by source and kind — the sparsity picture, in one table."""
    if labels.empty:
        return pd.DataFrame()
    g = (
        labels.groupby(["isoform", "source", "kind"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    totals = labels.groupby("isoform").size().rename("total")
    return g.merge(totals, on="isoform")


def to_wide(labels: pd.DataFrame, kinds=("interval",)) -> pd.DataFrame:
    """Pivot back to one row per compound, one column per endpoint (measured only)."""
    sub = labels[labels["kind"].isin(kinds)]
    return (
        sub.pivot_table(index=C.ID_COL, columns="endpoint", values="y", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
