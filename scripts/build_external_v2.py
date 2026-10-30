#!/usr/bin/env python
"""Rebuild the external table from the RAW AID 1851 dose-response, keeping the inactives.

    python scripts/build_external_v2.py --raw downloaded/datatable_1851.csv

WHAT WAS WRONG WITH artifacts/external_cyp.csv. It reported one pseudo-pIC50 per isoform
and, for every compound the screen called inactive, pinned that pseudo-pIC50 at the
censoring constant 4.2431:

    isoform   real AC50 values   pinned at 4.2431
    CYP1A2          58%                42%
    CYP2C9          47%                53%
    CYP2C19         56%                44%
    CYP2D6          33%                67%
    CYP3A4          53%                47%

Those rows were then fed to `ex_<iso>_pic50` as REGRESSION targets, so the trunk was
trained to emit one constant on up to two thirds of the external data. That is a
compression pressure, and it lands hardest exactly where the multi-head run failed:
CYP2D6 (67% pinned) fell to rho 0.4471 against v6's 0.4807 while three of four heads
collapsed to a predicted sd of 0.35-0.39.

WHAT THE LEADER DOES INSTEAD, in their words: "We keep `max_response` rather than its
pIC50: it is recorded at 100% coverage, so the 42,355 rows the screen called inactive
still carry signal." Our copy of the same file has 42,460 such rows and the column is
there -- we discarded it.

    Max_Response by outcome        n        mean      sd
      Active                    24069     -90.20   20.52
      Inactive                  42460      -5.29   15.21
      Inconclusive              19186     -30.65   58.67

A graded number, present for every row, that separates the classes by four standard
deviations. That is a far better auxiliary target than a constant.

WHAT THIS WRITES. One row per compound, and per isoform:

    <iso>_pic50      -Fit_LogAC50, ONLY where a real curve was fitted AND the curve is
                     an inhibition class (negative Fit_CurveClass). NCGC signs activation
                     curves positive; 2,530 of the 43,109 fits are activators and do not
                     belong in a potency head.
    <iso>_maxresp    Max_Response, sign-flipped so larger = more inhibition (matching the
                     pIC50 heads' direction) and winsorised to [-50, 150] -- the raw
                     column ranges to +1330 on activation artefacts.
    <iso>_curveclass the NCGC quality class, kept so a later run can filter on it
    <iso>_active     the screen's own call

NO pinned constants, and nothing is harmonised: these stay separate heads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PANEL = {"p450-cyp1a2": "CYP1A2", "p450-cyp2c9": "CYP2C9", "p450-cyp2c19": "CYP2C19",
         "p450-cyp2d6": "CYP2D6", "p450-cyp3a4": "CYP3A4"}
MR_CLIP = (-50.0, 150.0)


def inchikey_block(smiles):
    m = Chem.MolFromSmiles(str(smiles)) if isinstance(smiles, str) else None
    if m is None:
        return None, None
    try:
        return Chem.MolToInchiKey(m)[:14], Chem.MolToSmiles(m)
    except Exception:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="downloaded/datatable_1851.csv")
    ap.add_argument("--out", default="artifacts/external_cyp_v2.csv")
    a = ap.parse_args()

    d = pd.read_csv(a.raw, low_memory=False)
    d = d[d["PUBCHEM_RESULT_TAG"].astype(str).str.isdigit()].copy()
    d["iso"] = d["Panel Name"].map(PANEL)
    d = d[d["iso"].notna()]
    for c in ("Fit_LogAC50", "Fit_CurveClass", "Max_Response"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # one SMILES per CID, and its connectivity block
    cid_smi = (d.dropna(subset=["PUBCHEM_CID", "PUBCHEM_EXT_DATASOURCE_SMILES"])
                 .drop_duplicates("PUBCHEM_CID")
                 .set_index("PUBCHEM_CID")["PUBCHEM_EXT_DATASOURCE_SMILES"])
    print(f"{len(cid_smi)} unique CIDs; computing InChIKey blocks "
          f"(this is the slow step)...")
    blk, std = {}, {}
    for cid, smi in cid_smi.items():
        b, s = inchikey_block(smi)
        if b:
            blk[cid], std[cid] = b, s
    print(f"  {len(blk)} parsed, {len(cid_smi) - len(blk)} unparseable")

    d["block"] = d["PUBCHEM_CID"].map(blk)
    d = d[d["block"].notna()]

    out = pd.DataFrame({"block": sorted(d["block"].unique())}).set_index("block")
    out["smiles_std"] = out.index.map(
        d.drop_duplicates("block").set_index("block")["PUBCHEM_CID"].map(std))

    print(f"\n{'isoform':9s} {'pic50':>8s} {'activators':>11s} {'maxresp':>9s} "
          f"{'active':>7s}")
    for iso in sorted(set(PANEL.values())):
        s = d[d["iso"] == iso].drop_duplicates("block").set_index("block")
        inhib = s["Fit_CurveClass"] < 0                 # NCGC signs inhibition negative
        pic = np.where(inhib & s["Fit_LogAC50"].notna(), -s["Fit_LogAC50"], np.nan)
        out[f"{iso}_pic50"] = pd.Series(pic, index=s.index)
        out[f"{iso}_maxresp"] = (-s["Max_Response"]).clip(*MR_CLIP)
        out[f"{iso}_curveclass"] = s["Fit_CurveClass"]
        out[f"{iso}_active"] = (s["PUBCHEM_ACTIVITY_OUTCOME"] == "Active").astype(float)
        act = int(((~inhib) & s["Fit_LogAC50"].notna()).sum())
        print(f"{iso:9s} {int(np.isfinite(pic).sum()):8d} {act:11d} "
              f"{int(out[f'{iso}_maxresp'].notna().sum()):9d} "
              f"{int(out[f'{iso}_active'].sum()):7d}")

    # REGRESSION GUARD: the bug this file exists to fix must not come back.
    #
    # Threshold 0.20, not something tighter. qHTS reports AC50 on the COARSE GRID of its
    # 15-point concentration series, so a legitimate potency head is lumpy: the modal
    # CYP1A2 value is 4.8 (AC50 15.85 uM, mid-range) at 9%, then 4.9 at 8%, then 5.4 at
    # 5%. That is discretisation, not censoring. The bug looked completely different --
    # a single value at 42-67% sitting exactly at the assay ceiling -- so the guard also
    # checks the modal value is not parked at the top concentration.
    top_conc_pic50 = -np.log10(57.14e-6)                       # 4.243, the assay ceiling
    for iso in sorted(set(PANEL.values())):
        v = out[f"{iso}_pic50"].dropna()
        if not len(v):
            continue
        share = v.value_counts(normalize=True)
        mode, top = float(share.index[0]), float(share.iloc[0])
        assert top < 0.20, (f"{iso}_pic50 is {top:.0%} one value ({mode:.4f}) -- too "
                            f"lumpy even for the concentration grid")
        assert abs(mode - top_conc_pic50) > 0.05 or top < 0.05, (
            f"{iso}_pic50 piles {top:.0%} at the assay ceiling {top_conc_pic50:.4f} -- "
            f"the censoring constant is back in the potency head")
    out.reset_index().to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(out)} compounds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
