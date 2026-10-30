#!/usr/bin/env python
"""Assemble the multi-head training table: one column per ASSAY, never merged.

WHY THIS SHAPE. Our external-data attempts all failed the same way: we converted an
outside measurement into a pseudo-pIC50 and poured it into the scored column. That needs
a cross-assay correction we never had, and it washed out (+0.052 out-of-fold macro rho,
+0.004 blind).

The current leader (macro ST-RAE 0.4378, rank 1 of 81 —
https://supercowpowers.github.io/workbench/blogs/cyp_challenge/) does the opposite and
says why: "Every extra source measures something adjacent to the scored target, not
identical to it. Pooling any of them into the scored columns needs a cross-assay
correction; as separate heads none is required." One shared encoder, up to 18 heads, and
"only the four scored pIC50 heads are ever submitted; the rest exist to shape the
representation."

So nothing here is harmonised. Each assay keeps its own column, its own scale and its own
loss term. The auxiliary heads are gradient signal for the trunk and are thrown away at
prediction time.

THE HEADS.

    ch_<iso>_pic50   4   challenge dose-response pIC50            <- THE ONLY SCORED ONES
    ch_<iso>_lfc     4   challenge single-concentration log2fc, 4,376 compounds
    ex_<iso>_pic50   5   external pIC50, INCLUDING CYP2C19 which the challenge never
                         scores — an unscored isoform is free auxiliary chemistry
    ex_<iso>_maxresp 5   external qHTS max response, 100% coverage

18 heads over ~20,000 compounds. The external block is 16,233 compounds of which only 175
touch the challenge deck, so its contribution is breadth, not more labels on molecules we
already have — which is exactly the claim being tested.

WHY maxresp EXISTS, AND WHY THE FIRST RUN TIED WITHOUT IT. The original external file
reported one pseudo-pIC50 per isoform and pinned it at the censoring constant 4.2431 for
every compound the screen called inactive — 42% of CYP1A2 rows and 67% of CYP2D6 rows.
Fed to a REGRESSION head, that trains the trunk to emit a constant on two thirds of the
external data, and CYP2D6 is exactly the head that came back worse than the incumbent
(rho 0.4471 vs 0.4807) with three of four heads compressed to a predicted sd of 0.35-0.39.

`Max_Response` is the fix, and it is what the leader keeps: recorded for every row,
graded, and separating the screen's own classes by four standard deviations (Active
-90.2 +- 20.5, Inactive -5.3 +- 15.2). So the potency head now carries ONLY real curve
fits (modal value 6-10%, the qHTS concentration grid) and the inactives go to a head that
can represent them honestly instead of as a spike.

LEAKAGE GUARD, NON-NEGOTIABLE. Every external row whose InChIKey connectivity block
matches a BLIND compound is dropped before anything else happens, and `build_table`
reports the count. Matching on the connectivity block (first 14 characters) rather than
the full key means a salt, a stereoisomer or a different protonation state of a blind
compound cannot slip through as "new" chemistry.

WHAT IS NOT DONE HERE. No unit conversion, no assay-bias correction, no duplicate
reconciliation across sources — because with separate heads none of it is needed. The one
thing that IS shared is the molecule identity, and that is deliberate: the same compound
measured in two assays should reach the trunk as one graph with two labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from . import config as C, io

RDLogger.DisableLog("rdApp.*")

EXT_ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4")


def inchikey_block(smiles: str) -> str | None:
    """First 14 characters of the InChIKey — the connectivity layer.

    Salt, stereo and charge variants of one skeleton share it, which is what makes it the
    right key both for joining sources and for the leakage guard.
    """
    m = Chem.MolFromSmiles(str(smiles)) if isinstance(smiles, str) else None
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


def head_names(with_external: bool = True, with_screen: bool = True,
               with_maxresp: bool = True, with_tdi=True) -> list[str]:
    out = [f"ch_{i}_pic50" for i in C.ISOFORMS]          # scored, always first
    if with_screen:
        out += [f"ch_{i}_lfc" for i in C.ISOFORMS]
    if with_tdi:
        # The challenge's OWN second assay condition, not converted to anything. Section 70
        # added all four; Section 71 cut it to CYP3A4, the ONLY endpoint where the TDI arm
        # supplies compounds the scored file lacks (1,249 of them). On the other three it
        # supplies a duplicate reading of compounds already scored, which measured as
        # -0.026 rho on CYP1A2 and -0.014 on CYP2D6. `with_tdi` may be a bool or an
        # iterable of isoform names.
        out += [f"ch_{i}_tdi" for i in _tdi_isoforms(with_tdi)]
    if with_external:
        out += [f"ex_{i}_pic50" for i in EXT_ISOFORMS]
        if with_maxresp:
            out += [f"ex_{i}_maxresp" for i in EXT_ISOFORMS]
    return out


def _tdi_isoforms(with_tdi) -> tuple:
    """True -> CYP3A4 only (the Section 71 default). A list/tuple -> exactly those."""
    if with_tdi is True:
        return ("CYP3A4",)
    if not with_tdi:
        return ()
    bad = [i for i in with_tdi if i not in C.ISOFORMS]
    if bad:
        raise SystemExit(f"--tdi-isoforms: not scored isoforms: {bad}")
    return tuple(with_tdi)


def scored_head_index(names: list[str]) -> list[int]:
    return [names.index(f"ch_{i}_pic50") for i in C.ISOFORMS]


def build_table(external_csv: str | None = "artifacts/external_cyp_v2.csv",
                with_screen: bool = True,
                with_maxresp: bool = True,
                with_tdi=True,
                public_weight: float = 0.30,
                data_dir=None,
                verbose: bool = True) -> dict:
    """Returns dict(smiles, Y, W, names, is_challenge, ids).

    Y is (n_compounds, n_heads) with NaN wherever that compound has no measurement for
    that assay. W is the same shape: 1.0 for challenge rows, `public_weight` for external
    ones, 0 where Y is NaN. The leader uses --public-weight 0.30; that is the default here
    for the same reason, so external labels inform the trunk without outvoting 4,905
    challenge compounds with 16,145 of their own.
    """
    names = head_names(external_csv is not None, with_screen, with_maxresp, with_tdi)
    # KEY ON THE CONNECTIVITY BLOCK, NOT THE SMILES STRING. The challenge file and the
    # external file spell the same molecule differently (raw vs standardised SMILES), so
    # keying on the string leaves 175 structures present TWICE -- once as a challenge row
    # carrying a fold, once as an external row that is always in training. The model would
    # then see a validation compound's structure during training. Keying on the InChIKey
    # block collapses them into one row with both labels, which is also what the leader
    # means by "aligned on standardized InChIKey connectivity blocks".
    key_smiles: dict = {}          # block -> the SMILES we will build the graph from
    lab = pd.read_csv(C.artifact("labels.csv"))
    train = io.load("train_inhibition", data_dir)
    test = io.load("test", data_dir)
    test.columns = [c.strip() for c in test.columns]

    # ---- challenge compounds -------------------------------------------------------
    folds = pd.read_csv(C.artifact("folds.csv"))
    rows = {}                      # smiles -> {head: value}
    smi_of = dict(zip(folds[C.ID_COL], folds[C.SMILES_COL]))
    piv = lab[lab["kind"] != "left"].pivot_table(
        index=C.ID_COL, columns="isoform", values="y", aggfunc="mean")
    block_of_id = {}
    for mid, smi in smi_of.items():
        b = inchikey_block(smi)
        if b is None:
            continue
        block_of_id[mid] = b
        key_smiles.setdefault(b, smi)          # challenge spelling wins
        r = rows.setdefault(b, {})
        for iso in C.ISOFORMS:
            if iso in piv.columns and mid in piv.index and np.isfinite(piv.loc[mid, iso]):
                r[f"ch_{iso}_pic50"] = float(piv.loc[mid, iso])

    if with_screen:
        sc = io.load("single_conc", data_dir)
        g = sc.groupby([C.ID_COL, "enzyme"])["log2fc_estimate"].mean()
        for (mid, enz), v in g.items():
            if enz in C.ISOFORMS and mid in block_of_id and np.isfinite(v):
                # NOT converted to a pIC50. It is a different measurement; it gets a
                # different head. Sign flipped so "more inhibition" is larger, matching
                # the pIC50 heads' direction and making the trunk's job easier.
                rows.setdefault(block_of_id[mid], {})[f"ch_{enz}_lfc"] = -float(v)

    if with_tdi:
        # THE SAME ASSAY, A DIFFERENT CONDITION -- so it gets its own head and is never
        # converted. Nothing here needs a cross-assay correction, which is the whole
        # reason this module exists.
        #
        # LEAKAGE. A compound's TDI-condition curve is highly correlated with its scored
        # direct-inhibition curve, so handing the model one while scoring the other would
        # be a read-off. It is safe ONLY because train_cyp_mh.py holds out whole ROWS
        # (`va = fold == k; tr = ~va`), not individual cells: a validation compound's TDI
        # head is withheld along with its scored head. Do not change that to cell-level
        # masking without removing these heads first.
        tdi_isos = _tdi_isoforms(with_tdi)
        td = io.load("train_tdi", data_dir)
        td.columns = [c.strip() for c in td.columns]
        n_before, n_val = len(rows), 0
        for smi, sub in zip(td[C.SMILES_COL], td.to_dict("records")):
            b = inchikey_block(smi)
            if b is None:
                continue
            vals = {}
            for iso in tdi_isos:
                v = sub.get(f"{iso}_pIC50_TDI_condition")
                if v is not None and np.isfinite(v):
                    vals[f"ch_{iso}_tdi"] = float(v)
            # A TDI row carrying no label for any ENABLED head is not a compound, it is
            # dead weight on the trunk -- do not create a row for it.
            if not vals and b not in rows:
                continue
            key_smiles.setdefault(b, smi)      # challenge spelling already wins if present
            rows.setdefault(b, {}).update(vals)
            n_val += len(vals)
        if verbose:
            print(f"  TDI heads {tdi_isos}: {n_val} label values over {len(td)} compounds, "
                  f"{len(rows) - n_before} of them new to the challenge table")

    n_challenge = len(rows)
    challenge_blocks = set(rows)

    # ---- leakage guard, before any external row is used ------------------------------
    blind_blocks = set()
    for s in test[C.SMILES_COL]:
        b = inchikey_block(s)
        if b:
            blind_blocks.add(b)

    dropped = 0
    merged = 0
    if external_csv is not None:
        ext = pd.read_csv(external_csv)
        # v2 is keyed on `block` already; the old file carried a full `inchikey`.
        key = (ext["block"] if "block" in ext.columns
               else ext["inchikey"]).astype(str).str[:14]
        hit = key.isin(blind_blocks)
        dropped = int(hit.sum())
        ext, key = ext[~hit], key[~hit]
        smi_col = "smiles_std" if "smiles_std" in ext.columns else C.SMILES_COL
        # v2 columns are `<iso>_pic50` / `<iso>_maxresp`; the old file used `<iso>_pIC50`
        pic_col = {i: (f"{i}_pic50" if f"{i}_pic50" in ext.columns else f"{i}_pIC50")
                   for i in EXT_ISOFORMS}
        for b, r_ in zip(key, ext.itertuples(index=False)):
            smi = getattr(r_, smi_col, None) or getattr(r_, "SMILES", None)
            if not isinstance(smi, str) or not b:
                continue
            if b in challenge_blocks:
                merged += 1
            key_smiles.setdefault(b, smi)
            d = rows.setdefault(b, {})
            for iso in EXT_ISOFORMS:
                v = getattr(r_, pic_col[iso], np.nan)
                if v is not None and np.isfinite(v):
                    d[f"ex_{iso}_pic50"] = float(v)
                if with_maxresp:
                    v = getattr(r_, f"{iso}_maxresp", np.nan)
                    if v is not None and np.isfinite(v):
                        d[f"ex_{iso}_maxresp"] = float(v)

    # ---- matrix ---------------------------------------------------------------------
    blocks = list(rows)
    smiles = [key_smiles[b] for b in blocks]
    Y = np.full((len(blocks), len(names)), np.nan, np.float32)
    for i, b in enumerate(blocks):
        for h, v in rows[b].items():
            Y[i, names.index(h)] = v
    is_ch = np.array([b in challenge_blocks for b in blocks])
    W = np.where(np.isfinite(Y), 1.0, 0.0).astype(np.float32)
    ext_cols = [j for j, n in enumerate(names) if n.startswith("ex_")]
    W[:, ext_cols] *= public_weight

    if verbose:
        print(f"multi-head table: {len(smiles)} compounds x {len(names)} heads")
        print(f"  challenge compounds {n_challenge}   external-only "
              f"{len(smiles) - n_challenge}")
        print(f"  LEAKAGE GUARD: dropped {dropped} external rows whose InChIKey "
              f"connectivity block matches a blind compound")
        print(f"  merged into an existing challenge structure: {merged} "
              f"(these would have been DUPLICATE ROWS under SMILES-string keying, with "
              f"one copy leaking into training)")
        print(f"  labels {int(np.isfinite(Y).sum())}   public_weight {public_weight}")
        print(f"  {'head':18s} {'labels':>8s} {'mean':>8s} {'sd':>7s}")
        for j, n in enumerate(names):
            col = Y[:, j]; ok = np.isfinite(col)
            tag = "  <- SCORED" if n.startswith("ch_") and n.endswith("_pic50") else ""
            print(f"  {n:18s} {int(ok.sum()):8d} {col[ok].mean():8.3f} "
                  f"{col[ok].std():7.3f}{tag}")
    assert len(set(blocks)) == len(blocks), "duplicate structure in the table"
    return dict(smiles=smiles, blocks=blocks, Y=Y, W=W, names=names,
                is_challenge=is_ch, n_dropped=dropped)
