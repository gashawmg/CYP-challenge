#!/usr/bin/env python
r"""Train a tabular head on the cached representations. Seconds per configuration.

    python scripts/fit_tabular.py --model tabm --blocks rdkit2d ionisation --tag tabm_core
    python scripts/fit_tabular.py --model hgb  --blocks rdkit2d ionisation morgan --svd 512
    python scripts/fit_tabular.py --list                      # what can this env run?

Then score it, through the service that owns the labels:

    python scripts/eval_service.py score pred_<tag>.csv --dry-run

WHY THIS SHAPE
--------------
Smbatyan et al. (bioRxiv 2026.10.01.755314) reached 4th/104 on ExpansionRx and 1st/104 on
PXR with TabM, TabPFN, kernel regressors and MLPs over FROZEN representations -- not by
fine-tuning a graph network. The practical consequence on this project's hardware is the
point: a chemprop fold costs 10-25 minutes because the encoder is retrained each time;
a tabular head over cached features costs seconds. That is what makes a search over
configurations, by you or by an agent, actually possible.

TabM IS IMPLEMENTED HERE, NOT IMPORTED. It is a small idea -- an MLP whose weight
matrices are shared across k internal members, each member carrying its own rank-1 input
and output modulation and its own bias, so k predictors train for roughly the cost of one
and are averaged at inference. Roughly 60 lines of torch. Implementing it inline avoids
adding a dependency to an environment you have said is fragile; nothing in this file
installs anything, and models whose packages are missing are skipped and named.

THE HOLDOUT IS EXCLUDED, LOUDLY
-------------------------------
If artifacts/eval/holdout_ids.csv exists, those compounds are removed from training and
the count is printed. They still receive predictions -- that is how eval_service scores
them. Training on them would make every number this file produces meaningless, so
--ignore_holdout exists but shouts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

FEAT = Path("features")
TRAIN_CSV = "data/cyp-challenge-TRAIN_inhibition.csv"
SCREEN_CSV = "data/cyp-challenge-single-concentration-TRAIN.csv"
HOLDOUT_IDS = Path("artifacts/eval/holdout_ids.csv")

# Primary-screen hit rule, straight from the challenge tutorial (and cypkit/config.py).
# A compound that was SCREENED, did NOT hit, and never got a dose-response curve is a
# real measurement -- "no measurable activity" -- recorded as pIC50 below CENS_BOUND.
HIT_LOG2FC_MAX, HIT_FDR_MAX = -1.0, 0.05
CENS_BOUND = 4.0
ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
ID_COL, SMILES_COL = "Molecule_Name", "SMILES"


def ep(iso):
    return f"{iso}_pIC50_direct_inhibition"


# ------------------------------------------------------------------ TabM
def build_tabm(d_in, k, width, depth, dropout, device):
    """TabM: one MLP whose matrices are shared by k members, each with rank-1 modulation.

    A member's layer is  ((x * r_i) @ W) * s_i + b_i  -- W is shared, (r_i, s_i, b_i) are
    not. Each member therefore sees a differently scaled view of the input and produces a
    differently scaled output, which decorrelates them enough to behave like an ensemble,
    while the parameter count stays close to a single MLP's. Members are averaged at
    prediction time and the loss is summed over members at training time.
    """
    import torch
    from torch import nn

    class EnsLinear(nn.Module):
        def __init__(self, din, dout):
            super().__init__()
            self.W = nn.Linear(din, dout, bias=False)
            self.r = nn.Parameter(torch.ones(k, din))
            self.s = nn.Parameter(torch.ones(k, dout))
            self.b = nn.Parameter(torch.zeros(k, dout))
            # break the symmetry between members; without this they stay identical
            nn.init.normal_(self.r, 1.0, 0.1)
            nn.init.normal_(self.s, 1.0, 0.1)

        def forward(self, x):                      # x: (B, k, din)
            return self.W(x * self.r) * self.s + self.b

    class TabM(nn.Module):
        def __init__(self):
            super().__init__()
            layers, d = [], d_in
            for _ in range(depth):
                layers += [EnsLinear(d, width)]
                d = width
            self.layers = nn.ModuleList(layers)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(dropout)
            self.head = EnsLinear(width, 1)

        def forward(self, x):                      # x: (B, d_in)
            h = x.unsqueeze(1).expand(-1, k, -1)
            for lin in self.layers:
                h = self.drop(self.act(lin(h)))
            return self.head(h).squeeze(-1)        # (B, k)

    return TabM().to(device)


def fit_tabm(Xtr, ytr, Xva, yva, Xall, a, cens=None):
    """TabM with a censoring-aware objective.

    Observed rows take a Huber loss on the value. CENSORED rows take a ONE-SIDED hinge:
    predicting at or below the bound costs nothing, predicting above it pays the distance.
    That is deliberately the same shape as ST-RAE's dead zone, so the training objective
    and the scoring metric finally agree -- cypkit/labels.py records that mismatching them
    was a real bug in an earlier arm of this project.

    The censored rows carry no value, so they can only ever push predictions DOWN. They
    teach the model what an inactive compound looks like, which a curve-only model has
    never seen.
    """
    import torch
    from torch import nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    m = build_tabm(Xtr.shape[1], a.k, a.width, a.depth, a.dropout, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    T = lambda z: torch.tensor(z, dtype=torch.float32, device=dev)   # noqa: E731
    Xt, yt, Xv, yv = T(Xtr), T(ytr), T(Xva), T(yva)
    lossf = nn.HuberLoss(delta=0.5)

    has_c = cens is not None and len(cens["Xtr"])
    if has_c:
        Ct, Cv = T(cens["Xtr"]), T(cens["Xva"])
        bound = float(cens["bound"])
        w = float(a.cens_weight)

    def hinge(X):
        """mean over members and rows of relu(pred - bound) -- zero when already low."""
        return torch.clamp(m(X) - bound, min=0).mean()

    best, best_state, bad = np.inf, None, 0
    n, nc = len(Xt), len(Ct) if has_c else 0
    # censored rows are sampled proportionally, so a batch mixes both kinds
    cb = max(1, int(round(a.batch * nc / max(n, 1)))) if has_c else 0
    for epoch in range(a.epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            opt.zero_grad()
            pred = m(Xt[idx])
            loss = lossf(pred, yt[idx].unsqueeze(1).expand_as(pred))
            if has_c:
                cidx = torch.randint(0, nc, (cb,), device=dev)
                loss = loss + w * hinge(Ct[cidx])
            loss.backward(); opt.step()
        m.eval()
        with torch.inference_mode():
            vl = float(lossf(m(Xv).mean(1), yv))
            if has_c:
                vl += w * float(torch.clamp(m(Cv).mean(1) - bound, min=0).mean())
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    with torch.inference_mode():
        out = m(T(Xall)).mean(1).cpu().numpy()
    return out, epoch + 1


# ------------------------------------------------------------------ other models
def fit_sklearn(kind, Xtr, ytr, Xva, yva, Xall, a, cens=None):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.svm import SVR
    if kind == "hgb":
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          early_stopping=True, random_state=a.seed)
    elif kind == "rf":
        m = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=a.seed,
                                  min_samples_leaf=2)
    elif kind == "ridge":
        m = RidgeCV(alphas=np.logspace(-2, 4, 25))
    elif kind == "svr":
        m = SVR(C=10.0, epsilon=0.1, gamma="scale")
    else:
        raise ValueError(kind)
    if cens is not None and len(cens["Xtr"]):
        # APPROXIMATE. A squared/absolute loss cannot express "at most this value", so the
        # censored rows are imputed AT the bound and down-weighted. That over-penalises a
        # correct low prediction, which the hinge in fit_tabm does not. Use --model tabm
        # for the real thing; this exists so the tabular baselines see the same rows.
        Xtr = np.vstack([Xtr, cens["Xtr"]])
        ytr = np.concatenate([ytr, np.full(len(cens["Xtr"]), float(cens["bound"]))])
        sw = np.concatenate([np.ones(len(ytr) - len(cens["Xtr"])),
                             np.full(len(cens["Xtr"]), float(a.cens_weight))])
        try:
            m.fit(Xtr, ytr, sample_weight=sw)
            return m.predict(Xall), 0
        except TypeError:
            pass                                   # estimator has no sample_weight
    m.fit(Xtr, ytr)
    return m.predict(Xall), 0


def fit_tabpfn(Xtr, ytr, Xva, yva, Xall, a, cens=None):
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as e:
        raise ImportError("pip install tabpfn") from e
    import torch
    m = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu")
    m.fit(Xtr, ytr)
    return m.predict(Xall), 0


MODELS = {
    "tabm": (fit_tabm, "torch", "TabM -- k-member weight-shared MLP (the paper's workhorse)"),
    "hgb": (lambda *x, **kw: fit_sklearn("hgb", *x, **kw), "sklearn", "histogram gradient boosting"),
    "rf": (lambda *x, **kw: fit_sklearn("rf", *x, **kw), "sklearn", "random forest"),
    "ridge": (lambda *x, **kw: fit_sklearn("ridge", *x, **kw), "sklearn", "ridge with generalised CV"),
    "svr": (lambda *x, **kw: fit_sklearn("svr", *x, **kw), "sklearn", "RBF support-vector regression"),
    "tabpfn": (fit_tabpfn, "tabpfn", "TabPFN v2 tabular foundation model (<=10k rows)"),
}


# ------------------------------------------------------------------ data
def load_blocks(names, svd, seed):
    idx = pd.read_csv(FEAT / "index.csv")
    mats, used = [], []
    for b in names:
        f = FEAT / f"{b}.npy"
        if not f.exists():
            print(f"  block '{b}' not cached -- run build_features.py first; skipped")
            continue
        X = np.load(f)
        if X.shape[0] != len(idx):
            print(f"  block '{b}' has {X.shape[0]} rows, index has {len(idx)}; skipped")
            continue
        mats.append(X); used.append((b, X.shape[1]))
    if not mats:
        raise SystemExit("no usable feature blocks")
    X = np.hstack(mats).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # drop dead columns before anything else -- a constant column is pure noise to a
    # tree and an invitation to divide by zero everywhere else
    keep = X.std(axis=0) > 1e-8
    X = X[:, keep]
    note = f"{sum(n for _, n in used)} cols -> {X.shape[1]} after dropping constants"

    if svd and X.shape[1] > svd:
        from sklearn.decomposition import TruncatedSVD
        # fitted on ALL compounds including the blind set. That is unsupervised and uses
        # no labels, so it is not leakage; it is the standard transductive setting and it
        # is what the paper's "SVD reductions of log-count fingerprints" describes.
        X = TruncatedSVD(n_components=svd, random_state=seed).fit_transform(X)
        note += f", SVD -> {svd}"
    return idx, X.astype(np.float32), note


def censored_ids(iso):
    """Compounds that were screened, did not hit, and have no curve for this endpoint.

    These are NOT missing data. The compound was tested and did nothing, which bounds its
    pIC50 above: the only thing we know is y <= CENS_BOUND. Dropping them throws away
    69-71% of the evidence on three endpoints, and Section 49 showed that is where the
    0.12 macro ST-RAE separating this project's simple models from its production blend
    actually lives -- three unrelated architectures trained on curves alone all landed at
    0.755, while the blend trained on these rows reached 0.639.

    Derived from the raw challenge files, not from artifacts/labels.csv, so this script
    stays self-contained. Verified to reproduce that file's `left` set exactly (2700 /
    2134 / 2883 / 707), the only difference being the sub-4.0 curve rows, which are a
    separate decision (see train_cyp_clean.py's SUBFLOOR_BOUND).
    """
    f = Path(SCREEN_CSV)
    if not f.exists():
        return set(), f"{SCREEN_CSV} not found -- censored rows unavailable"
    sc = pd.read_csv(f); sc.columns = [c.strip() for c in sc.columns]
    s_ = sc[sc["enzyme"] == iso]
    if not len(s_):
        return set(), f"no screen rows for {iso}"
    hit = (s_["log2fc_estimate"] <= HIT_LOG2FC_MAX) & (s_["log2fc_fdr"] <= HIT_FDR_MAX)
    per = s_.assign(hit=hit).groupby(ID_COL)["hit"].max()   # a hit at ANY concentration
    return set(per[~per].index), ""


def scaffold_groups(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            out.append(f"__bad__{s}"); continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:                                            # noqa: BLE001
            sc = ""
        out.append(sc or f"__acyclic__{Chem.MolToSmiles(m)}")
    return out


def activity_folds(groups, y, n_folds, seed, n_bins=4):
    """Scaffold-disjoint folds, balanced on the activity distribution rather than count."""
    bins = pd.qcut(pd.Series(y), n_bins, labels=False, duplicates="drop").to_numpy()
    n_bins = int(np.nanmax(bins)) + 1
    g = {}
    for i, s in enumerate(groups):
        g.setdefault(s, []).append(i)
    rng = np.random.default_rng(seed)
    order = sorted(g.values(), key=lambda v: (-len(v), rng.random()))
    target = np.array([[(bins == b).sum() / n_folds for b in range(n_bins)]] * n_folds)
    have = np.zeros((n_folds, n_bins))
    fold = np.full(len(y), -1, int)
    for grp in order:
        add = np.zeros(n_bins)
        for i in grp:
            add[bins[i]] += 1
        cost = ((have + add - target) ** 2).sum(1) - ((have - target) ** 2).sum(1)
        k = int(np.argmin(cost))
        have[k] += add
        for i in grp:
            fold[i] = k
    return fold


# ------------------------------------------------------------------ metrics / layout
def st_rae(p, lo, hi, const):
    def cost(q):
        return float(np.sum(np.where(np.isfinite(hi), np.clip(q - hi, 0, None), 0.0)
                          + np.where(np.isfinite(lo), np.clip(lo - q, 0, None), 0.0)))
    b = cost(np.full(len(p), const))
    return cost(p) / b if b > 0 else np.nan


def report(p, y, lo, hi):
    s, t = pd.Series(p), pd.Series(y)
    vp = float(np.var(p))
    return dict(n=len(y), st_rae=st_rae(p, lo, hi, float(np.mean(y))),
                mae=float(np.mean(np.abs(p - y))),
                r2=float(1 - np.mean((y - p) ** 2) / np.var(y)),
                rho=float(s.corr(t, method="spearman")),
                slope=float(np.cov(p, y, bias=True)[0, 1] / vp) if vp > 1e-12 else np.nan)


W = 76
HEAD = (f"\n{'':11s}{'------- held out -------':^33s}  {'--- trained on ---':^18s}\n"
        f"{'fold':>5s}{'n':>6s}{'ST-RAE':>9s}{'MAE':>8s}{'R2':>8s}{'rho':>8s}"
        f"  {'MAE':>8s}{'R2':>8s}{'gap':>7s}\n" + "-" * W)


def row(tag, mv, mt=None):
    def f(v, w):
        return f"{v:>{w}.4f}" if v is not None and np.isfinite(v) else f"{'-':>{w}s}"
    right = (f(mt["mae"], 8) + f(mt["r2"], 8)
             + f"{(mv['mae'] - mt['mae']) / max(mv['mae'], 1e-9):>+6.0%} " if mt
             else f"{'':>23s}")
    return (f"{tag:>5s}{mv['n']:>6d}" + f(mv["st_rae"], 9) + f(mv["mae"], 8)
            + f(mv["r2"], 8) + f(mv["rho"], 8) + "  " + right).rstrip()


# ------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tabm", choices=list(MODELS))
    p.add_argument("--blocks", nargs="*", default=["rdkit2d", "ionisation"])
    p.add_argument("--isoforms", nargs="*", default=list(ISOFORMS))
    p.add_argument("--tag", default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--svd", type=int, default=0, help="reduce the concatenated matrix")
    p.add_argument("--use_censored", action="store_true",
                   help="ALSO train on the screened-but-inactive compounds (2,700 / "
                        "2,134 / 2,883 / 707 extra rows) with a one-sided loss that "
                        "charges nothing for predicting them below the bound. Scoring is "
                        "unchanged -- only curve compounds are ever scored -- so results "
                        "stay comparable with runs made without it")
    p.add_argument("--cens_bound", type=float, default=CENS_BOUND,
                   help="pIC50 these compounds are known to be at or below")
    p.add_argument("--cens_weight", type=float, default=1.0,
                   help="weight on the censored term relative to the observed loss")
    p.add_argument("--standardise", action="store_true", default=True)
    p.add_argument("--list", action="store_true", help="models and cached blocks, then exit")
    p.add_argument("--ignore_holdout", action="store_true",
                   help="DO NOT USE: trains on the eval holdout and invalidates every "
                        "score the harness will report")
    # TabM
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    a = p.parse_args()
    a.tag = a.tag or f"{a.model}_{'_'.join(a.blocks)}"

    if a.list:
        print(f"{'model':10s}{'needs':10s}{'status':9s} what it is")
        for n, (_, req, desc) in MODELS.items():
            try:
                __import__(req); s = "ready"
            except ImportError:
                s = "MISSING"
            print(f"{n:10s}{req:10s}{s:9s} {desc}")
        print(f"\ncached blocks in {FEAT}/:")
        mf = FEAT / "manifest.json"
        if mf.exists():
            for k, v in json.loads(mf.read_text()).items():
                print(f"  {k:14s} {tuple(v['shape'])}  {v['description']}")
        else:
            print("  none -- run scripts/build_features.py")
        return 0

    fn, req, desc = MODELS[a.model]
    try:
        __import__(req)
    except ImportError:
        print(f"model '{a.model}' needs `{req}`, which is not installed. "
              f"`--list` shows what is.")
        return 1

    idx, X, note = load_blocks(a.blocks, a.svd, a.seed)
    tr = pd.read_csv(TRAIN_CSV); tr.columns = [c.strip() for c in tr.columns]
    pos = {m: i for i, m in enumerate(idx[ID_COL])}

    held = set()
    if HOLDOUT_IDS.exists() and not a.ignore_holdout:
        held = set(pd.read_csv(HOLDOUT_IDS)[ID_COL])
    elif a.ignore_holdout:
        print("!! --ignore_holdout: training on the eval holdout. Every score the "
              "harness reports for this file is meaningless.\n")

    print("=" * W)
    print(f"  {a.tag}")
    print("=" * W)
    print(f"  model      {a.model} -- {desc}")
    print(f"  features   {'+'.join(a.blocks)}   {note}")
    print(f"  holdout    {len(held)} compounds excluded from training"
          + ("" if held else "   (none found -- run eval_service.py init)"))

    groups_all = scaffold_groups(idx[SMILES_COL].tolist())
    oof_store, pred_store, summary = {}, {}, []

    for iso in a.isoforms:
        col = ep(iso)
        d = tr[tr[col].notna()]
        fitd = d[~d[ID_COL].isin(held)]
        ri = np.array([pos[m] for m in fitd[ID_COL]])
        y = fitd[col].to_numpy(float)
        lo = fitd[col + "_conf_low"].to_numpy(float)
        hi = fitd[col + "_conf_high"].to_numpy(float)
        grp = [groups_all[i] for i in ri]
        fold = activity_folds(grp, y, a.folds, a.seed)

        # ---- censored rows: screened, no hit, no curve -------------------------------
        cri, cfold, cmsg = np.array([], int), np.array([], int), ""
        if a.use_censored:
            cids, cmsg = censored_ids(iso)
            cids = {m for m in cids if m not in held and m in pos
                    and m not in set(fitd[ID_COL])}
            if cids:
                cri = np.array(sorted(pos[m] for m in cids))
                # A censored row joins the fold its SCAFFOLD already went to, so a
                # censored analogue of a validation compound never sits in training.
                # Scaffolds unseen among the curve rows are spread round-robin.
                where = {}
                for j, g in enumerate(grp):
                    where.setdefault(g, fold[j])
                rng = np.random.default_rng(a.seed)
                cfold = np.array([where.get(groups_all[i], int(rng.integers(a.folds)))
                                  for i in cri], int)

        Xs = X
        if a.standardise:
            rows = np.concatenate([ri, cri]) if len(cri) else ri
            mu, sd = X[rows].mean(0), X[rows].std(0)
            Xs = (X - mu) / np.where(sd > 1e-8, sd, 1.0)

        print(f"\n  {iso}   {len(d)} curves, {len(fitd)} after holdout, "
              f"{len(set(grp))} scaffolds"
              + (f", + {len(cri)} censored (<= {a.cens_bound:.1f}, weight "
                 f"{a.cens_weight:g})" if len(cri) else
                 (f"   [censored: {cmsg}]" if cmsg else "")))
        print(HEAD)
        oof = np.full(len(fitd), np.nan)
        allp, tr_ms, va_ms, t0 = [], [], [], time.time()
        for k in range(a.folds):
            va = np.where(fold == k)[0]; trn = np.where(fold != k)[0]
            ym, ys = y[trn].mean(), (y[trn].std() or 1.0)
            cens = None
            if len(cri):
                ct, cv = cri[cfold != k], cri[cfold == k]
                cens = dict(Xtr=Xs[ct], Xva=Xs[cv] if len(cv) else Xs[ct][:1],
                            bound=(a.cens_bound - ym) / ys)   # bound in standardised units
            out, eps = fn(Xs[ri[trn]], (y[trn] - ym) / ys,
                          Xs[ri[va]], (y[va] - ym) / ys, Xs, a, cens=cens)
            out = out * ys + ym
            allp.append(out)
            oof[va] = out[ri[va]]
            mv = report(oof[va], y[va], lo[va], hi[va])
            mt = report(out[ri[trn]], y[trn], lo[trn], hi[trn])
            tr_ms.append(mt); va_ms.append(mv)
            print(row(str(k + 1), mv, mt))
        pooled = report(oof, y, lo, hi)
        mean_tr = {kk: float(np.mean([m[kk] for m in tr_ms])) for kk in tr_ms[0]}
        mean_tr["n"] = int(mean_tr["n"])
        print("-" * W)
        print(row("pool", pooled, mean_tr))
        print(f"  {a.folds} folds in {time.time() - t0:.1f}s   slope {pooled['slope']:.3f}")

        oof_store[iso] = pd.Series(oof, index=fitd[ID_COL].to_numpy())
        pred_store[iso] = np.mean(np.vstack(allp), axis=0)
        summary.append(dict(iso=iso, **{kk: pooled[kk] for kk in
                                        ("n", "st_rae", "mae", "r2", "rho", "slope")}))

    print("\n" + "=" * W)
    print(f"  SUMMARY   {a.tag}   (out-of-fold, holdout compounds NOT in training)")
    print("=" * W)
    print(f"{'endpoint':11s}{'n':>6s}{'ST-RAE':>9s}{'MAE':>8s}{'R2':>8s}{'rho':>8s}"
          f"{'slope':>8s}")
    print("-" * W)
    for r in summary:
        print(f"{r['iso']:11s}{r['n']:>6d}{r['st_rae']:>9.4f}{r['mae']:>8.4f}"
              f"{r['r2']:>8.4f}{r['rho']:>8.4f}{r['slope']:>8.3f}")
    print("-" * W)
    print(f"{'macro':11s}{'':>6s}"
          + "".join(f"{float(np.mean([r[kk] for r in summary])):>{w}.4f}"
                    for kk, w in (("st_rae", 9), ("mae", 8), ("r2", 8), ("rho", 8))))

    ids = sorted(set().union(*[set(s.index) for s in oof_store.values()]))
    o = pd.DataFrame({ID_COL: ids})
    for iso, s in oof_store.items():
        o[ep(iso)] = o[ID_COL].map(s)
    o.to_csv(f"oof_{a.tag}.csv", index=False)

    sub = idx[[SMILES_COL, ID_COL]].copy()
    for iso in ISOFORMS:
        if iso in pred_store:
            sub[ep(iso)] = pred_store[iso]
    sub.to_csv(f"pred_{a.tag}.csv", index=False)
    print(f"\nwrote oof_{a.tag}.csv and pred_{a.tag}.csv "
          f"({len(sub)} rows: train + blind)")
    print(f"score it:  python scripts/eval_service.py score pred_{a.tag}.csv --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
