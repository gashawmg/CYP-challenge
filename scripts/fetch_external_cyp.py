#!/usr/bin/env python
"""Pull public CYP inhibition data and prepare it as auxiliary training labels.

    python scripts/fetch_external_cyp.py            # on a machine with internet
    python scripts/fetch_external_cyp.py --from-csv downloaded/AID1851.csv

WHY THIS SCRIPT EXISTS. The top of the pIC50 leaderboard explains 62% of blind-set
variance (R^2 0.62, Spearman 0.76). Our best out-of-fold R^2 on our data-RICHEST
endpoint is 0.58, and on the blind set we manage 0.15. The TDI board tells the same
story: leaders sit at d' 1.20-1.66 (AUC 0.80-0.88) against our 0.93 (0.745). A gap
that size is not an architecture gap. We have 4,905 compounds; CYP inhibition is one
of the most heavily measured endpoints in public cheminformatics.

The obvious source is Veith et al. 2009 (Nat Biotechnol 27:1050), the NCGC qHTS
CYP450 panel — PubChem AID 1851 and its children — roughly 12,000-17,000 compounds
assayed against CYP1A2, 2C9, 2C19, 2D6 and 3A4. Four of those five are exactly our
endpoints, and the assay chemistry (P450-Glo luminescence) is the same family as the
challenge's. That is 2.5-3.5x our training set.

TWO THINGS TO SETTLE BEFORE USING ANY OF IT
-------------------------------------------
1. RULES. The leaderboard carries a "Proprietary Data" column, so the organisers
   clearly anticipate external data — but check the challenge rules yourself before
   submitting a model trained on it. This script does not decide that for you.

2. LEAKAGE. Some of the 750 blind-test compounds may have public measurements. Using
   those would be reading the answer key out of PubChem, not modelling. This script
   ALWAYS removes external rows whose standardised InChIKey (first block, so
   stereochemistry and salt form do not hide a match) collides with a blind-test
   compound, and reports how many it dropped. Do not disable that.

WHAT IT WRITES. `artifacts/external_cyp.csv` with one row per compound:
SMILES, InChIKey, and one column per isoform. The Veith panel is reported as a binary
inhibitor call, so these are auxiliary CLASSIFICATION labels, not pIC50 — useful as
extra multitask heads or for pretraining the message-passing trunk, not as direct
regression targets. If you obtain the raw AC50 table instead, pass --from-csv and the
parser will convert AC50 (uM) to pIC50 where the column exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TDC_SETS = {
    "CYP1A2": "CYP1A2_Veith",
    "CYP2C9": "CYP2C9_Veith",
    "CYP2D6": "CYP2D6_Veith",
    "CYP3A4": "CYP3A4_Veith",
    "CYP2C19": "CYP2C19_Veith",   # not scored, but a strong auxiliary head
}


def standardise(smiles: str):
    """Canonical SMILES + InChIKey first block, or (None, None)."""
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem.MolStandardize.rdMolStandardize import (  # noqa: PLC0415
        LargestFragmentChooser, Uncharger)

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, None
        mol = Uncharger().uncharge(LargestFragmentChooser().choose(mol))
        return Chem.MolToSmiles(mol, canonical=True), Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None, None


PUBCHEM_CSV = ("https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?"
               "query=download&record_type=datatable&actvty=all&"
               "response_type=save&aid={aid}")


def download_aid(aid: int, dest: Path) -> Path:
    """Fetch a PubChem assay data table directly - no PyTDC, no dependency fight.

    PyTDC pins an old scikit-learn with no Python 3.13 wheel, so pip tries to compile
    it from source; and letting it resolve dependencies can downgrade the
    scikit-learn/torch a working chemprop environment depends on. This route needs
    only `requests`.

    NOTE: this endpoint could not be reached from the sandbox this script was written
    in, so the URL is unverified here. If it fails, download the data table by hand
    from https://pubchem.ncbi.nlm.nih.gov/bioassay/1851 (Download -> CSV) and pass
    --from-csv.
    """
    import requests  # noqa: PLC0415

    url = PUBCHEM_CSV.format(aid=aid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    r = requests.get(url, timeout=300, stream=True)
    r.raise_for_status()
    n = 0
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk); n += len(chunk)
    print(f"  saved {dest} ({n/1e6:.1f} MB)")
    if n < 10_000:
        raise SystemExit(
            f"{dest} is only {n} bytes - PubChem probably returned an error page.\n"
            "Download the data table by hand from "
            "https://pubchem.ncbi.nlm.nih.gov/bioassay/1851 and use --from-csv.")
    return dest


def from_tdc(sets):
    """PyTDC handles the PubChem download and the train/valid/test concatenation."""
    try:
        from tdc.single_pred import ADME  # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            "PyTDC not installed. Either:\n"
            "    pip install PyTDC\n"
            "or download the AID 1851 data table from PubChem by hand and use\n"
            "    --from-csv <file>")
    frames = {}
    for iso, name in sets.items():
        print(f"  fetching {name} ...", flush=True)
        df = ADME(name=name).get_data()
        cols = {c.lower(): c for c in df.columns}
        smi = cols.get("drug", cols.get("smiles"))
        y = cols.get("y", cols.get("label"))
        frames[iso] = df[[smi, y]].rename(columns={smi: "SMILES", y: iso})
        print(f"    {len(df)} rows, {df[y].mean()*100:.1f}% positive")
    return frames


def from_csv(path):
    """Parse a PubChem assay data table. AC50 columns become pIC50 where present."""
    raw = pd.read_csv(path, low_memory=False)
    smi_col = next((c for c in raw.columns if c.lower() in ("smiles", "pubchem_ext_datasource_smiles")), None)
    if smi_col is None:
        raise SystemExit(f"no SMILES column in {path}; columns: {list(raw.columns)[:20]}")
    frames = {}
    for iso in TDC_SETS:
        ac = [c for c in raw.columns if iso.lower() in c.lower() and "ac50" in c.lower()]
        act = [c for c in raw.columns if iso.lower() in c.lower() and "activity" in c.lower()]
        if ac:
            v = pd.to_numeric(raw[ac[0]], errors="coerce")
            # AC50 in uM -> pIC50 = 6 - log10(AC50)
            frames[iso] = pd.DataFrame({"SMILES": raw[smi_col],
                                        iso: 6.0 - np.log10(v.where(v > 0))}).dropna()
            print(f"  {iso}: {len(frames[iso])} AC50 values -> pIC50")
        elif act:
            frames[iso] = pd.DataFrame({"SMILES": raw[smi_col],
                                        iso: raw[act[0]]}).dropna()
            print(f"  {iso}: {len(frames[iso])} activity calls (binary)")
    if not frames:
        raise SystemExit("no per-isoform columns recognised in that file")
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-csv", default=None,
                    help="a PubChem AID 1851 data table, instead of PyTDC")
    ap.add_argument("--download-aid", type=int, default=None,
                    help="fetch a PubChem assay data table directly (try 1851); "
                         "avoids PyTDC and its dependency conflicts")
    ap.add_argument("--isoforms", nargs="*", default=list(TDC_SETS))
    ap.add_argument("--test-csv", default="data/cyp-challenge-TEST-BLINDED.csv")
    ap.add_argument("--train-csv", default="data/cyp-challenge-TRAIN_inhibition.csv")
    ap.add_argument("--out", default="artifacts/external_cyp.csv")
    args = ap.parse_args()

    sets = {k: v for k, v in TDC_SETS.items() if k in args.isoforms}
    print("Fetching external CYP data")
    if args.download_aid:
        path = download_aid(args.download_aid, Path("downloaded") / f"AID{args.download_aid}.csv")
        frames = from_csv(path)
    elif args.from_csv:
        frames = from_csv(args.from_csv)
    else:
        frames = from_tdc(sets)

    print("\nStandardising and merging on InChIKey...")
    merged = None
    for iso, df in frames.items():
        std = df["SMILES"].map(standardise)
        df = df.assign(smiles_std=[s for s, _ in std], inchikey=[k for _, k in std])
        df = df.dropna(subset=["smiles_std", "inchikey"])
        df = df.groupby("inchikey", as_index=False).agg(
            {"smiles_std": "first", iso: "mean"})
        merged = df if merged is None else merged.merge(df, on=["inchikey", "smiles_std"],
                                                        how="outer")
    print(f"  {len(merged)} unique compounds")

    # --- leakage guard --------------------------------------------------
    test = pd.read_csv(args.test_csv)
    test.columns = [c.strip() for c in test.columns]
    tsmi = next(c for c in test.columns if c.lower() == "smiles")
    test_keys = {k for _, k in (standardise(s) for s in test[tsmi]) if k}
    hit = merged["inchikey"].isin(test_keys)
    print(f"\nLEAKAGE GUARD: {int(hit.sum())} external compound(s) match a BLIND TEST "
          f"compound by InChIKey -- removing them.")
    if hit.any():
        print("  (their public measurements would be the answer key, not a prediction)")
    merged = merged[~hit].reset_index(drop=True)

    train = pd.read_csv(args.train_csv)
    train.columns = [c.strip() for c in train.columns]
    trsmi = next(c for c in train.columns if c.lower() == "smiles")
    train_keys = {k for _, k in (standardise(s) for s in train[trsmi]) if k}
    overlap = merged["inchikey"].isin(train_keys)
    print(f"Overlap with the challenge TRAINING set: {int(overlap.sum())} compounds "
          f"({100*overlap.mean():.1f}%) -- kept, but they add no new molecules.")
    merged["in_challenge_train"] = overlap

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"\nwritten {args.out}")
    print(f"{'isoform':10s} {'labelled':>9s} {'new molecules':>14s}")
    for iso in sets:
        if iso in merged.columns:
            n = int(merged[iso].notna().sum())
            new = int((merged[iso].notna() & ~overlap).sum())
            print(f"{iso:10s} {n:9d} {new:14d}")
    print("\nCheck the challenge rules on external data before training on this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
