#!/usr/bin/env python
r"""t-SNE of each endpoint's training compounds against the blind set.

    python scripts/tsne_domain.py
    python scripts/tsne_domain.py --perplexity 40 --outdir figures

THE QUESTION THIS ANSWERS, AND THE ONE IT DOES NOT. "Is the blind set somewhere our
training data is not?" The measured answer is NO: the blind compounds tile the same
regions on every endpoint. What differs between the four panels is not the blind set --
it is the same 750 compounds every time -- but which compounds carry a dose-response
curve for that isoform.

The second and third figures ask the question that actually matters. Mean |pIC50
difference| between two training compounds is FLAT against similarity on CYP1A2, CYP2C9
and CYP2D6 -- about 0.65 log units whether the pair is at Tanimoto 0.35 or 0.55 -- and
falls properly only on CYP3A4 (1.14 -> 0.48). So the useful picture is not where the blind
compounds sit but whether the training neighbourhoods they sit IN have a consistent
activity to lend them.

METRIC. t-SNE runs on JACCARD distance over Morgan bits, not Euclidean -- Tanimoto is the
similarity every other part of this project uses, and embedding under a different metric
would produce a picture that corresponds to none of our other numbers.

COLOUR, AND WHY IT IS WHAT IT IS. All three figures were checked with the data-viz
validator rather than by eye:

  fig 1  categorical blue #2a78d6 / orange #eb6834 -- worst all-pairs CVD dE 24.7 (protan),
         32.7 (tritan) against a >= 8 target, and both clear 3:1 on the surface. Marker
         SHAPE also differs (filled vs open), so identity never rests on colour alone.

  fig 2  sequential blue, restricted to steps 400-700. The full 100-700 ramp is for
         heatmap cells, where the lightest step may recede into the surface. On a SCATTER
         every point has to be seen, and steps 100-350 measure 1.29-2.91:1 against this
         surface -- the first render of this figure had low-activity compounds effectively
         invisible. 400 (#3987e5) is the lightest step that clears 3:1.

  fig 3  diverging blue <-> red about a NEUTRAL GREY MIDPOINT AT 1.0, which is the point
         where a compound's nearest labelled neighbours agree no better than five
         compounds drawn at random. Diverging is right here because that midpoint is a
         real zero, not a convenience. Poles pass all-pairs CVD dE 21.6 (protan) / 34.5
         (tritan).

WHAT A t-SNE PLOT CANNOT TELL YOU. Distances between well-separated clusters are not
meaningful, cluster sizes are not meaningful, and the layout changes with perplexity and
seed. Every number printed on or under a panel is computed from the fingerprints directly,
never from the embedding.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

ISOS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")

TRAIN_C = "#2a78d6"     # categorical slot 1
BLIND_C = "#eb6834"     # categorical slot 2
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
CONTEXT = "#d6d5d0"     # recessive grey for context marks

# sequential blue, steps 400-700 only: every step clears 3:1 on this surface
SEQ = ["#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# diverging blue <-> red, neutral grey midpoint (the documented pair)
DIV = ["#0d366b", "#1c5cab", "#2a78d6", "#86b6ef", "#f0efec",
       "#f2a3a2", "#e34948", "#b32423", "#7d1817"]

# mean |x1-x2| for two draws from a normal with sd s is 2s/sqrt(pi) = 1.1284 s.
# That is the "neighbours are no better than random" reference for the k-NN spread.
RANDOM_PAIR = 2.0 / np.sqrt(np.pi)


def subtitle(fig, text, width=104):
    """Wrapped deck line under the title.

    A single fig.text() runs straight off the canvas -- matplotlib does not wrap, and the
    first render of this figure lost the end of both subtitles to the right edge with no
    error. Wrapping here rather than hand-breaking the string keeps the sentences editable.
    """
    for i, ln in enumerate(textwrap.wrap(text, width)):
        fig.text(0.055, 0.949 - i * 0.017, ln, color=INK2, fontsize=9, ha="left")


def fingerprints(smiles):
    from rdkit import Chem, RDLogger                                  # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator                     # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        X.append(np.frombuffer(gen.GetFingerprint(m).ToBitString().encode(),
                               "u1") - ord("0"))
        keep.append(i)
    return np.asarray(X, dtype=bool), np.asarray(keep)


def _sim_block(Xa, Xb, i, block):
    a = Xa[i:i + block].astype(np.float32)
    b = Xb.astype(np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def nn_stats(Xa, Xb, block=256):
    out = np.empty(len(Xa), np.float32)
    for i in range(0, len(Xa), block):
        out[i:i + block] = _sim_block(Xa, Xb, i, block).max(axis=1)
    return out


def local_spread(Xb, Xt, yt, k=5, block=256):
    """Mean |y - y| among the k nearest TRAINING compounds to each blind compound.

    This is what decides whether a neighbourhood can answer a question at all. A blind
    compound whose five nearest labelled compounds disagree by a log unit cannot be
    predicted to better than that by any model reading this fingerprint.
    """
    out = np.empty(len(Xb), np.float32)
    for i in range(0, len(Xb), block):
        sim = _sim_block(Xb, Xt, i, block)
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        for j in range(len(idx)):
            v = yt[idx[j]]
            out[i + j] = float(np.mean(np.abs(v[:, None] - v[None, :])))
    return out


# A label sitting directly on 2,000 scattered points is not readable at any ink colour;
# it needs to sit ON the surface, not in the cloud.
NOTE_BOX = dict(facecolor=SURFACE, edgecolor="none", alpha=0.82, pad=3.0)


def panel_frame(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()

    import matplotlib                                                 # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                                   # noqa: PLC0415
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: PLC0415
    from sklearn.manifold import TSNE                                 # noqa: PLC0415

    cmap_seq = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    cmap_div = LinearSegmentedColormap.from_list("div_br", DIV)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv("artifacts/labels.csv"); lab.columns = [c.strip() for c in lab.columns]
    test = pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv")
    test.columns = [c.strip() for c in test.columns]
    Xb_all, _ = fingerprints(test["SMILES"].tolist())
    print(f"blind: {len(Xb_all)} of {len(test)} parsed")

    emb, meta = {}, {}
    for iso in ISOS:
        t = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
        t = t.reset_index(drop=True)
        Xt, keep_t = fingerprints(t["SMILES"].tolist())
        y = t["y"].to_numpy(float)[keep_t]

        nn_b = nn_stats(Xb_all, Xt)
        sp = local_spread(Xb_all, Xt, y, k=a.k)
        ref = RANDOM_PAIR * float(np.std(y))      # disagreement expected from random draws
        ratio = sp / max(ref, 1e-9)

        X = np.vstack([Xt, Xb_all])
        print(f"  {iso}: t-SNE on {len(X)} compounds ({len(Xt)} train, "
              f"{len(Xb_all)} blind)...", flush=True)
        Z = TSNE(n_components=2, metric="jaccard", init="random",
                 perplexity=a.perplexity, random_state=a.seed,
                 max_iter=1000).fit_transform(X.astype(np.float32))
        emb[iso] = (Z[:len(Xt)], Z[len(Xt):], y, ratio)
        meta[iso] = dict(n_tr=len(Xt), nn_b=float(np.median(nn_b)),
                         spread=float(np.median(sp)), ref=ref,
                         ratio=float(np.median(ratio)),
                         worse=float(np.mean(ratio >= 1.0)),
                         far=float(np.mean(nn_b < 0.4)))

    # ---- figure 1: do the two sets occupy the same space? -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), facecolor=SURFACE)
    for ax, iso in zip(axes.ravel(), ISOS):
        zt, zb, _, _ = emb[iso]
        panel_frame(ax, iso)
        ax.scatter(zt[:, 0], zt[:, 1], s=9, c=TRAIN_C, alpha=0.45, linewidths=0,
                   label=f"training ({meta[iso]['n_tr']})", rasterized=True)
        ax.scatter(zb[:, 0], zb[:, 1], s=16, facecolors="none", edgecolors=BLIND_C,
                   linewidths=0.7, alpha=0.85, label=f"blind ({len(zb)})", rasterized=True)
        ax.text(0.02, 0.02,
                f"median NN to labelled {meta[iso]['nn_b']:.3f}\n"
                f"{meta[iso]['far']:.0%} of blind below 0.40",
                transform=ax.transAxes, fontsize=9, color=INK2, va="bottom",
                bbox=NOTE_BOX)
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Blind compounds sit inside the training chemistry, on every endpoint",
                 color=INK, fontsize=15, x=0.055, ha="left", y=0.975)
    subtitle(fig, "t-SNE on Jaccard distance over Morgan r=2 bits. Filled = training, "
                  "open ring = blind, so the two are told apart by shape as well as "
                  "colour. Cluster sizes and between-cluster distances are not "
                  "meaningful; the printed numbers come from the fingerprints.")
    fig.tight_layout(rect=[0, 0, 1, 0.915])
    p = out / "tsne_train_vs_blind.png"; fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig); print(f"wrote {p}")

    # ---- figure 2: is the activity field smooth where they sit? -----------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), facecolor=SURFACE)
    for ax, iso in zip(axes.ravel(), ISOS):
        zt, zb, y, _ = emb[iso]
        panel_frame(ax, iso)
        ax.scatter(zb[:, 0], zb[:, 1], s=7, c=CONTEXT, alpha=0.9, linewidths=0,
                   rasterized=True)
        lo, hi = np.percentile(y, [2, 98])
        s = ax.scatter(zt[:, 0], zt[:, 1], s=11, c=y, cmap=cmap_seq, vmin=lo, vmax=hi,
                       alpha=0.95, linewidths=0, rasterized=True)
        cb = fig.colorbar(s, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("training pIC50", color=INK2, fontsize=9)
        cb.ax.tick_params(colors=INK2, labelsize=8); cb.outline.set_edgecolor(GRID)
    fig.suptitle("Where activity is smooth in chemical space, and where it is speckled",
                 color=INK, fontsize=15, x=0.055, ha="left", y=0.975)
    subtitle(fig, "Training coloured by pIC50 (ramp restricted to steps that clear 3:1 on "
                  "this surface, so no compound is invisible); blind compounds in grey. A "
                  "smooth field means similarity carries activity -- only CYP3A4 shows "
                  "one.")
    fig.tight_layout(rect=[0, 0, 1, 0.915])
    p = out / "tsne_activity_field.png"; fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig); print(f"wrote {p}")

    # ---- figure 3: neighbourhood disagreement, per blind compound ---------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), facecolor=SURFACE)
    norm = TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=1.7)
    for ax, iso in zip(axes.ravel(), ISOS):
        zt, zb, _, ratio = emb[iso]
        panel_frame(ax, iso)
        ax.scatter(zt[:, 0], zt[:, 1], s=6, c=CONTEXT, alpha=0.85, linewidths=0,
                   rasterized=True)
        s = ax.scatter(zb[:, 0], zb[:, 1], s=22, c=ratio, cmap=cmap_div, norm=norm,
                       alpha=0.95, linewidths=0.25, edgecolors=SURFACE, rasterized=True)
        cb = fig.colorbar(s, ax=ax, fraction=0.045, pad=0.02,
                          ticks=[0.4, 0.7, 1.0, 1.3, 1.6])
        cb.set_label("disagreement / random", color=INK2, fontsize=9)
        cb.ax.tick_params(colors=INK2, labelsize=8); cb.outline.set_edgecolor(GRID)
        m = meta[iso]
        ax.text(0.02, 0.02,
                f"median {m['spread']:.2f} log units = {m['ratio']:.2f}x random\n"
                f"{m['worse']:.0%} of blind at or worse than random",
                transform=ax.transAxes, fontsize=9, color=INK2, va="bottom",
                bbox=NOTE_BOX)
    fig.suptitle("How much the nearest labelled neighbours disagree, per blind compound",
                 color=INK, fontsize=15, x=0.055, ha="left", y=0.975)
    subtitle(fig, f"Each blind compound coloured by the mean |pIC50 difference| among its "
                  f"{a.k} nearest labelled compounds, divided by what {a.k} RANDOM "
                  f"compounds would give. Grey midpoint = 1.0 is the point where "
                  f"similarity has bought nothing; blue is informative, red is worse than "
                  f"chance. Training in grey for context.")
    fig.tight_layout(rect=[0, 0, 1, 0.915])
    p = out / "tsne_neighbour_disagreement.png"; fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig); print(f"wrote {p}")

    print(f"\n  {'endpoint':9s} {'n lab':>6s} {'med NN':>7s} {'<0.40':>6s} "
          f"{'k-NN disagree':>13s} {'random ref':>11s} {'ratio':>6s} {'>= random':>10s}")
    for iso in ISOS:
        m = meta[iso]
        print(f"  {iso:9s} {m['n_tr']:6d} {m['nn_b']:7.3f} {m['far']:6.0%} "
              f"{m['spread']:13.2f} {m['ref']:11.2f} {m['ratio']:6.2f} {m['worse']:10.0%}")
    print("\n  ratio < 1 means the nearest labelled neighbours agree better than random "
          "ones.\n  It is the fraction of the structure-activity signal these "
          "fingerprints actually carry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
