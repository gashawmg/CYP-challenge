#!/usr/bin/env python
"""Pull the Veith CYP450 panel straight from PubChem. No new dependencies.

    python scripts/fetch_pubchem_cyp.py

Uses only the Python standard library plus pandas and rdkit, which your chemprop
environment already has. Nothing is installed, nothing is upgraded, nothing in
`admet_gpu` is touched.

WHY NOT PyTDC. It pins an old scikit-learn with no Python 3.13 wheel, so pip tries to
compile it from source; and letting it resolve dependencies can quietly downgrade the
scikit-learn or torch a working chemprop install depends on. Not worth the risk for a
one-off data pull.

A BETTER PRIZE THAN PyTDC, INCIDENTALLY. The TDC "Veith" datasets are binarised
inhibitor / non-inhibitor calls. The PubChem source carries the qHTS **Potency**
column in micromolar, so this script produces real pIC50 REGRESSION targets for the
external compounds — the same quantity the challenge scores — falling back to the
binary activity outcome only where no potency was fitted. That is a strictly stronger
auxiliary signal.

HOW IT FINDS THE DATA. Your probe settled this: the classic datatable CGI answers for
AID 1851, and that one table already carries every isoform as its own column. So the
default route is a SINGLE request to that table — no child-AID discovery, which was
the fragile part (400s from a bad qualifier, then 503 throttling). `--use-children`
keeps the old per-isoform route as a fallback.

THE TABLE IS LONG, NOT WIDE. Your first run settled this too: AID 1851 has no
per-isoform columns at all. It is one row per compound x panel member — ~85,600 rows,
17,000-odd compounds x 5 isoforms — with the isoform in a `Panel Name` column and a
single shared `Potency` / `PUBCHEM_ACTIVITY_OUTCOME` pair. So the job is a pivot. Two
things fall out of that: `PUBCHEM_EXT_DATASOURCE_SMILES` is right there, so the
CID->SMILES round trips are gone; and the panel names are full HGNC ("cytochrome
P450, family 2, subfamily C, polypeptide 19"), which contain no "CYP2C19" substring
at all and are rewritten before matching.

PubChem also puts a RESULT_TYPE annotation row ("FLOAT", "INT", "STRING") between the
header and the data, so every column arrives as text and the declared dtype tells you
nothing — values are coerced with `to_numeric` and the branch is taken on what the
coercion produced, not on what the dtype claims.

INACTIVES ARE KEPT, AS CENSORED DATA. The panel's top concentration is 57.14 uM, so
an "Inactive" call means pIC50 < 4.24 — a real left-censored measurement. Each isoform
gets three columns: `_pIC50` (fitted, or the 4.24 floor), `_active` (binary), and
`_censored` (1 where the pIC50 is the floor). That matters here specifically: our
neural models train only on fitted curves and sit 24-30% BELOW the floor on known
inactives, which is most of what the dense proxy penalises them for. Use them as a
Tobit-style censored term, as a classification head, or drop them — this script does
not decide.

EVERY DOWNLOAD IS CACHED under `downloaded/`, so a failure part-way through costs you
only the remaining files, and a re-run is instant. A 200 response whose body is a
PubChem fault page is detected and discarded rather than cached and replayed forever.

I COULD NOT TEST THE NETWORK CALLS. The sandbox this was written in is blocked from
PubChem (403 at the proxy), so the parsing is tested against a synthetic table with
the annotation row in place and the URL comes from your own probe. Every step
validates what it got back and fails loudly; if one does, the printed message names
the manual alternative.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
SUMMARY_AID = 1851
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
PAUSE = 1.0           # PubChem throttles hard; 503 = PUGREST.ServerBusy
CACHE = Path("downloaded")


def get(url: str, dest: Path | None = None, binary: bool = False, tries: int = 3):
    """Cached GET with backoff. Returns text (or bytes)."""
    if dest and dest.exists() and dest.stat().st_size > 0:
        cached = dest.read_bytes()
        # A cached ERROR page is worse than no cache: it makes every later run fail
        # the same way for a reason that is no longer true. PubChem returns faults as
        # JSON with a "Fault" key, so treat those as a miss and re-fetch.
        if b'"Fault"' in cached[:2000] or b"PUGREST." in cached[:2000]:
            print(f"    (discarding cached error page {dest})")
            dest.unlink(missing_ok=True)
        else:
            return cached if binary else cached.decode("utf-8", errors="replace")
    last = None
    for k in range(tries):
        try:
            time.sleep(PAUSE)
            req = urllib.request.Request(url, headers={"User-Agent": "cyp-challenge/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if dest:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            return data if binary else data.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            # A 4xx is the server rejecting THIS url; retrying changes nothing.
            # Only 5xx and transport errors are worth a second attempt.
            if 400 <= e.code < 500:
                raise RuntimeError(f"HTTP {e.code} for {url}") from e
            last = e
            print(f"    attempt {k+1}/{tries}: HTTP {e.code}")
            time.sleep(2 ** k)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"    attempt {k+1}/{tries} failed: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(2 ** k)
    raise RuntimeError(f"could not fetch {url}: {last}")


VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"


def _aids_from_json(txt):
    """Any AID list, whatever shape PubChem wrapped it in."""
    try:
        j = json.loads(txt)
    except Exception:
        return []
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.upper() == "AID":
                    found.extend(v if isinstance(v, list) else [v])
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(j)
    out = []
    for a in found:
        try:
            out.append(int(a))
        except (TypeError, ValueError):
            pass
    return out


def child_aids(summary: int) -> list[int]:
    """The per-isoform assays underneath the summary AID.

    PubChem exposes this several ways and the right one varies by record, so try a
    list rather than betting on a single qualifier. `aids_type=component` returns
    HTTP 400 for this record -- reachable, but the wrong parameter -- which is why
    the first version of this script failed at the very first call.

    Last resort: the summary assay's own description text enumerates the panel
    members, so regex `AID 1234` out of it.
    """
    attempts = [
        ("plain /aids", f"{BASE}/assay/aid/{summary}/aids/JSON"),
        ("aids_type=doseresponse", f"{BASE}/assay/aid/{summary}/aids/JSON?aids_type=doseresponse"),
        ("aids_type=component", f"{BASE}/assay/aid/{summary}/aids/JSON?aids_type=component"),
        ("PUG-View record", f"{VIEW}/data/assay/{summary}/JSON"),
        ("description text", f"{BASE}/assay/aid/{summary}/description/JSON"),
    ]
    for name, url in attempts:
        try:
            txt = get(url, CACHE / f"discover_{summary}_{re.sub(r'[^a-z0-9]+','_',name.lower())}.json")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:24s} {str(e)[:70]}")
            continue
        ids = [a for a in _aids_from_json(txt) if a != summary]
        if not ids:
            ids = [int(m) for m in re.findall(r"AID[\s:=]*?(\d{3,7})", txt, re.I)
                   if int(m) != summary]
        ids = sorted(set(ids))
        if ids:
            print(f"  {name:24s} -> {len(ids)} candidate AID(s): {ids[:15]}")
            return ids
        print(f"  {name:24s} responded but contained no AIDs")

    raise SystemExit(
        "Could not enumerate the child assays automatically.\n"
        "Open https://pubchem.ncbi.nlm.nih.gov/bioassay/1851 and read the five child\n"
        "AIDs off the page (one per isoform), then:\n"
        "    python scripts/fetch_pubchem_cyp.py --aids <a> <b> <c> <d> <e>\n"
        "The script identifies which isoform each one is from its own description,\n"
        "so the order does not matter.")


def identify(aid: int) -> str | None:
    """Which isoform does this assay measure? Read its own description."""
    try:
        txt = get(f"{BASE}/assay/aid/{aid}/description/JSON",
                  CACHE / f"desc_{aid}.json")
    except Exception:
        return None
    blob = txt.upper().replace("-", "").replace(" ", "").replace("_", "")
    # PubChem titles say "cytochrome P450 2C19", not "CYP2C19". Normalise both
    # spellings to CYP<family> before matching, or every assay comes back None.
    blob = blob.replace("CYTOCHROMEP450", "CYP").replace("P450", "CYP")
    # longest first, so CYP2C19 is not swallowed by CYP2C1 / CYP2C9
    for iso in sorted(ISOFORMS, key=len, reverse=True):
        if iso in blob:
            return iso
    return None


def parse_assay(csv_path: Path, iso: str) -> pd.DataFrame:
    """CID -> pIC50 from the qHTS Potency column, else a binary activity call."""
    df = pd.read_csv(csv_path, low_memory=False)
    cid = next((c for c in df.columns if c.upper() == "PUBCHEM_CID"), None)
    if cid is None:
        raise SystemExit(f"{csv_path} has no PUBCHEM_CID column: {list(df.columns)[:15]}")
    df = df[pd.to_numeric(df[cid], errors="coerce").notna()].copy()
    df[cid] = df[cid].astype(np.int64)

    pot = next((c for c in df.columns
                if re.search(r"potency|ac50|ic50", str(c), re.I)), None)
    if pot is not None:
        v = pd.to_numeric(df[pot], errors="coerce")
        out = pd.DataFrame({"cid": df[cid], f"{iso}_pIC50": 6.0 - np.log10(v.where(v > 0))})
        out = out.dropna()
        print(f"    {iso}: {len(out)} potency values from '{pot}' -> pIC50 "
              f"[{out[f'{iso}_pIC50'].min():.2f}, {out[f'{iso}_pIC50'].max():.2f}]")
    else:
        out = pd.DataFrame(columns=["cid", f"{iso}_pIC50"])
        print(f"    {iso}: no potency column found")

    oc = next((c for c in df.columns if c.upper() == "PUBCHEM_ACTIVITY_OUTCOME"), None)
    if oc is not None:
        m = df[oc].astype(str).str.strip().str.lower()
        b = pd.DataFrame({"cid": df[cid],
                          f"{iso}_active": m.map({"active": 1.0, "inactive": 0.0})}).dropna()
        print(f"    {iso}: {len(b)} activity calls ({100*b[f'{iso}_active'].mean():.1f}% active)")
        out = out.merge(b, on="cid", how="outer")
    return out.groupby("cid", as_index=False).mean()


CGI = "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi"
DATATABLE = (CGI + "?query=download&record_type=datatable&actvty=all"
                   "&response_type=save&aid={aid}")


def datatable(aid: int) -> pd.DataFrame:
    """AID 1851's own summary table: every isoform, one request.

    This is the route the probe confirmed working. It skips child-AID discovery
    entirely, which is worth doing because discovery is the fragile part -- the
    qualifier that works varies by record and two of the three PUG-REST spellings
    return HTTP 400 for this one.

    (`record_type=concise` is NOT a valid option here; PubChem answers
    "PCGET.Unimplemented". Only `datatable` works.)
    """
    path = CACHE / f"datatable_{aid}.csv"
    get(DATATABLE.format(aid=aid), path, binary=True)
    size = path.stat().st_size
    print(f"  {path} ({size/1e6:.1f} MB)")
    if size < 50_000:
        raise SystemExit(
            f"{path} is only {size} bytes -- probably an error page, not data.\n"
            f"Open {DATATABLE.format(aid=aid)} in a browser, save the CSV, and pass\n"
            "    --from-datatable <file>")
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df)} rows x {len(df.columns)} columns")
    print(f"  columns: {list(df.columns)}")
    return df


# PubChem's top tested concentration in the Veith panel is 57.14 uM, so an
# "Inactive" call means pIC50 < 6 - log10(57.14) = 4.24. That is a real, LEFT-CENSORED
# measurement, not a missing value -- and it is exactly the population our neural
# models never saw (they train only on fitted curves, which is why they sit
# 24-30% below the floor on known inactives). Kept as an explicit floor value plus
# a censor flag so the training script can use a Tobit-style loss, treat them as a
# classification head, or drop them -- its choice, not this script's.
CENSOR_UM = 57.14
CENSOR_PIC50 = round(6.0 - np.log10(CENSOR_UM), 4)   # 4.2429

PANEL_COLS = ("Panel Name", "Panel Target", "Panel ID", "PUBCHEM_ASSAYDATA_COMMENT")


def _panel_to_iso(series: pd.Series) -> pd.Series:
    """Map a panel-member label onto one of our isoforms, or NA.

    Same normalisation as `identify()`: PubChem writes "cytochrome P450 2C19", not
    "CYP2C19". Longest isoform first, and already-assigned rows are never
    overwritten, so CYP2C19 is not captured by the CYP2C9 pattern.
    """
    s = series.astype(str).str.upper()
    # PubChem often uses the full HGNC name: "cytochrome P450, family 1, subfamily A,
    # polypeptide 2" is CYP1A2 and contains the string "CYP1A2" nowhere at all.
    s = s.str.replace(
        r"FAMILY\s*(\d+)\s*,?\s*SUBFAMILY\s*([A-Z])\s*,?\s*POLYPEPTIDE\s*(\d+)",
        r"CYP\1\2\3", regex=True)
    s = (s.str.replace(r"[-_,\s]+", "", regex=True)
         .str.replace("CYTOCHROMEP450", "CYP", regex=False)
         .str.replace("P450", "CYP", regex=False))
    out = pd.Series([None] * len(series), index=series.index, dtype=object)
    for iso in sorted(ISOFORMS, key=len, reverse=True):
        hit = out.isna() & s.str.contains(iso, regex=False, na=False)
        out[hit] = iso
    return out


def parse_long(df: pd.DataFrame, panel_col: str) -> pd.DataFrame:
    """AID 1851 is LONG, not wide: one row per compound x panel member.

    17,000-odd compounds x 5 isoforms ~= 85,600 rows, with the isoform in a `Panel
    Name` column and a single shared `Potency` / `PUBCHEM_ACTIVITY_OUTCOME` pair. So
    the job is a pivot, not a column hunt. `PUBCHEM_EXT_DATASOURCE_SMILES` is in the
    table too, which removes the ~115 CID->SMILES requests entirely.
    """
    iso_of = _panel_to_iso(df[panel_col])
    vc = df[panel_col].astype(str).value_counts()
    print(f"  panel column '{panel_col}' -> {len(vc)} distinct values")
    for v, n in vc.head(12).items():
        print(f"    {str(v)[:56]:58s} {n:7d}  -> {_panel_to_iso(pd.Series([v]))[0] or '(unmapped)'}")
    if iso_of.isna().all():
        raise SystemExit(
            f"none of the '{panel_col}' values name a CYP isoform. Values above.")
    print(f"  {int(iso_of.notna().sum())} of {len(df)} rows mapped to an isoform")

    cid = next((c for c in df.columns if c.upper() == "PUBCHEM_CID"), None)
    sid = next((c for c in df.columns if c.upper() == "PUBCHEM_SID"), None)
    smi = next((c for c in df.columns if "SMILES" in c.upper()), None)
    pot = next((c for c in df.columns if str(c).strip().lower() == "potency"), None) \
        or next((c for c in df.columns if re.search(r"potency|ac50|ic50", str(c), re.I)), None)
    oc = next((c for c in df.columns if c.upper() == "PUBCHEM_ACTIVITY_OUTCOME"), None)
    print(f"  columns in use: cid={cid} sid={sid} smiles={smi} potency={pot} outcome={oc}")
    if cid is None and sid is None:
        raise SystemExit("no PUBCHEM_CID or PUBCHEM_SID column to key on")

    work = pd.DataFrame({"iso": iso_of})
    # key on CID where there is one, SID otherwise -- some qHTS rows are substances
    # with no compound record, and throwing them away would lose real measurements
    k = pd.Series(pd.NA, index=df.index, dtype=object)
    if cid is not None:
        c = pd.to_numeric(df[cid], errors="coerce")
        k = c.map(lambda v: None if pd.isna(v) else f"cid:{int(v)}")
    if sid is not None:
        s = pd.to_numeric(df[sid], errors="coerce")
        k = k.where(k.notna(), s.map(lambda v: None if pd.isna(v) else f"sid:{int(v)}"))
    work["key"] = k
    if smi is not None:
        work["SMILES"] = df[smi]
    work["potency_um"] = pd.to_numeric(df[pot], errors="coerce") if pot else np.nan
    work["outcome"] = (df[oc].astype(str).str.strip().str.lower() if oc else "")
    work = work[work["iso"].notna() & work["key"].notna()]
    n_nocid = int(work["key"].str.startswith("sid:").sum())
    if n_nocid:
        print(f"  {n_nocid} rows keyed on SID (no compound record)")

    frames = []
    for iso in ISOFORMS:
        sub = work[work["iso"] == iso]
        if sub.empty:
            print(f"  {iso}: no rows")
            continue
        active = sub["outcome"].map({"active": 1.0, "inactive": 0.0})
        p = 6.0 - np.log10(sub["potency_um"].where(sub["potency_um"] > 0))
        # inactive with no fitted curve -> left-censored at the top concentration
        cens = p.isna() & (active == 0.0)
        p = p.where(~cens, CENSOR_PIC50)
        f = pd.DataFrame({"key": sub["key"].to_numpy(),
                          f"{iso}_pIC50": p.to_numpy(),
                          f"{iso}_active": active.to_numpy(),
                          f"{iso}_censored": cens.astype(float).to_numpy()})
        f = f.groupby("key", as_index=False).mean()
        fitted = int((f[f"{iso}_pIC50"].notna() & (f[f"{iso}_censored"] == 0)).sum())
        print(f"  {iso}: {len(f)} compounds | {fitted} fitted pIC50 | "
              f"{int((f[f'{iso}_censored'] > 0).sum())} censored at {CENSOR_PIC50} | "
              f"{100*f[f'{iso}_active'].mean(skipna=True):.1f}% active")
        frames.append(f)
    if not frames:
        raise SystemExit("no isoform produced any rows")

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="key", how="outer")
    if "SMILES" in work.columns:
        smiles = (work.dropna(subset=["SMILES"]).drop_duplicates("key")[["key", "SMILES"]])
        merged = merged.merge(smiles, on="key", how="left")
        print(f"  SMILES carried straight from the table for "
              f"{int(merged['SMILES'].notna().sum())} of {len(merged)} compounds")
    merged["cid"] = merged["key"]
    return merged.drop(columns=["key"])


def parse_datatable(df: pd.DataFrame) -> pd.DataFrame:
    """Pull per-isoform potency / activity columns out of the summary table.

    Column naming in PubChem summary tables is not fully predictable, so match on
    the isoform appearing anywhere in the column name (longest isoform first, so
    CYP2C19 is not captured by the CYP2C9 pattern) and then on what kind of value
    the column holds. Every recognised column is printed, so a miss is visible
    rather than silent.

    AID 1851 turned out NOT to be wide: it is one row per compound x panel member,
    with the isoform in a `Panel Name` column. That shape is detected here and sent
    to `parse_long`; the wide path below still serves single-isoform child assays.
    """
    for pc in PANEL_COLS:
        col = next((c for c in df.columns if str(c).strip().lower() == pc.lower()), None)
        if col is not None and _panel_to_iso(df[col]).notna().any():
            print(f"  long format detected ({len(df)} rows); pivoting on '{col}'")
            return parse_long(df, col)

    cid = next((c for c in df.columns if c.upper() == "PUBCHEM_CID"), None)
    if cid is None:
        raise SystemExit(f"no PUBCHEM_CID column. Columns: {list(df.columns)}")
    df = df[pd.to_numeric(df[cid], errors="coerce").notna()].copy()
    df[cid] = df[cid].astype(np.int64)
    print(f"  {len(df)} data rows after dropping header/annotation rows")

    out = pd.DataFrame({"cid": df[cid]})
    used = []
    for iso in sorted(ISOFORMS, key=len, reverse=True):
        pat = iso.replace("CYP", "").lower()          # e.g. "2c19"
        cols = [c for c in df.columns
                if re.search(rf"(cyp\s*)?{pat}\b", str(c), re.I)
                and not any(c in u for u in used)]
        if not cols:
            print(f"  {iso}: no matching column")
            continue
        pot = next((c for c in cols if re.search(r"potency|ac50|ic50", str(c), re.I)), None)
        act = next((c for c in cols if re.search(r"activ|outcome|score", str(c), re.I)), None)
        if pot:
            v = pd.to_numeric(df[pot], errors="coerce")
            out[f"{iso}_pIC50"] = 6.0 - np.log10(v.where(v > 0))
            n = int(out[f"{iso}_pIC50"].notna().sum())
            print(f"  {iso}: pIC50 from '{pot}' ({n} values)")
            used.append(pot)
        if act:
            a = df[act]
            # Do NOT branch on the DECLARED dtype. Two things defeat it:
            #   * pandas 3 gives text columns a dedicated `str` dtype, not `object`,
            #     so `dtype == object` was False and Active/Inactive fell down the
            #     numeric path;
            #   * PubChem datatables carry a RESULT_TYPE annotation row ("INT",
            #     "FLOAT", "STRING") above the data, so even a genuinely numeric
            #     column is read as text and `is_numeric_dtype` stays False after
            #     that row is dropped.
            # Coerce first and decide from what the coercion actually produced.
            v = pd.to_numeric(a, errors="coerce")
            m = a.astype(str).str.strip().str.lower().map(
                {"active": 1.0, "inactive": 0.0, "active agonist": 1.0,
                 "active inhibitor": 1.0, "inconclusive": np.nan,
                 "unspecified": np.nan})
            if v.notna().any():
                if re.search(r"score", str(act), re.I):
                    # PubChem activity SCORE runs 0-100; clipping it to [0,1] would
                    # turn every score above 1 into a positive call.
                    num = (v / 100.0).clip(0, 1)
                else:
                    num = v.clip(0, 1)
            else:
                num = pd.Series(np.nan, index=a.index, dtype=float)
            # numeric where the coercion worked, the Active/Inactive map elsewhere
            out[f"{iso}_active"] = num.where(num.notna(), m)
            n = int(out[f"{iso}_active"].notna().sum())
            print(f"  {iso}: activity from '{act}' ({n} calls)")
            used.append(act)
        if not pot and not act:
            print(f"  {iso}: columns {cols} matched but held no usable values")
    keep = [c for c in out.columns if c != "cid"]
    if not keep:
        raise SystemExit(
            "No per-isoform columns recognised. Paste the column list printed above "
            "and the matcher can be adjusted.")
    return out.dropna(subset=keep, how="all").groupby("cid", as_index=False).mean()


def cid_to_smiles(cids, chunk=150) -> pd.DataFrame:
    """CID -> canonical SMILES, in chunks, cached per chunk."""
    cids = sorted({int(c) for c in cids})
    print(f"  resolving SMILES for {len(cids)} CIDs "
          f"({(len(cids)+chunk-1)//chunk} requests)")
    frames = []
    for i in range(0, len(cids), chunk):
        part = cids[i:i + chunk]
        dest = CACHE / "smiles" / f"chunk_{i:06d}.csv"
        # SMILES property name changed across PubChem revisions; try both.
        for prop in ("CanonicalSMILES", "SMILES", "IsomericSMILES"):
            try:
                txt = get(f"{BASE}/compound/cid/{','.join(map(str, part))}/property/{prop}/CSV",
                          dest)
                d = pd.read_csv(pd.io.common.StringIO(txt))
                col = next((c for c in d.columns if "smiles" in c.lower()), None)
                if col:
                    frames.append(d.rename(columns={"CID": "cid", col: "SMILES"})[["cid", "SMILES"]])
                    break
            except Exception:
                dest.unlink(missing_ok=True)
                continue
        else:
            print(f"    WARNING: chunk at {i} failed for every property name; skipped")
        if (i // chunk) % 20 == 0 and i:
            print(f"    {i}/{len(cids)}")
    if not frames:
        raise SystemExit("no SMILES resolved - check network access to PubChem")
    return pd.concat(frames, ignore_index=True).drop_duplicates("cid")


def standardise(smiles):
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--use-children", action="store_true",
                    help="go via the per-isoform child assays instead of the "
                         "AID 1851 summary data table (slower, more fragile)")
    ap.add_argument("--from-datatable", default=None,
                    help="a data table CSV already downloaded by hand")
    ap.add_argument("--aids", type=int, nargs="*", default=None,
                    help="child AIDs, with --use-children, if discovery fails")
    ap.add_argument("--summary-aid", type=int, default=SUMMARY_AID)
    ap.add_argument("--test-csv", default="data/cyp-challenge-TEST-BLINDED.csv")
    ap.add_argument("--train-csv", default="data/cyp-challenge-TRAIN_inhibition.csv")
    ap.add_argument("--out", default="artifacts/external_cyp.csv")
    args = ap.parse_args()

    if not args.use_children:
        print(f"Downloading the AID {args.summary_aid} data table (all isoforms, one request)")
        df = (pd.read_csv(args.from_datatable, low_memory=False)
              if args.from_datatable else datatable(args.summary_aid))
        merged = parse_datatable(df)
        print(f"\n{len(merged)} unique CIDs with at least one label")
        return finish(merged, args)

    print(f"Discovering the CYP panel under AID {args.summary_aid}")
    aids = args.aids or child_aids(args.summary_aid)

    print("\nIdentifying each assay from its own description")
    assay = {}
    for a in aids:
        iso = identify(a)
        print(f"  AID {a}: {iso or 'not a CYP panel member - skipped'}")
        if iso and iso not in assay:
            assay[iso] = a
    missing = [i for i in ISOFORMS if i not in assay]
    if missing:
        print(f"  NOTE: no assay identified for {missing}")
    if not assay:
        raise SystemExit("no CYP assays identified; pass --aids explicitly")

    print("\nDownloading assay data")
    tables = []
    for iso, a in assay.items():
        print(f"  AID {a} ({iso})")
        path = CACHE / f"aid_{a}.csv"
        get(f"{BASE}/assay/aid/{a}/CSV", path, binary=True)
        tables.append(parse_assay(path, iso))

    merged = tables[0]
    for t in tables[1:]:
        merged = merged.merge(t, on="cid", how="outer")
    print(f"\n{len(merged)} unique CIDs across the panel")

    return finish(merged, args)


def finish(merged, args):
    # The AID 1851 data table ships PUBCHEM_EXT_DATASOURCE_SMILES, so the CID->SMILES
    # round trips are only needed when a route did not supply structures.
    have = ("SMILES" in merged.columns and merged["SMILES"].notna().sum() >
            0.5 * len(merged))
    if have:
        n_missing = int(merged["SMILES"].isna().sum())
        print(f"\nSMILES already in the table for {len(merged) - n_missing} compounds"
              + (f"; dropping {n_missing} without one" if n_missing else ""))
        merged = merged.dropna(subset=["SMILES"]).reset_index(drop=True)
    else:
        cids = [int(c.split(":")[1]) for c in merged["cid"].astype(str)
                if str(c).startswith("cid:")] or merged["cid"]
        smi = cid_to_smiles(cids)
        smi["cid"] = "cid:" + smi["cid"].astype(str)
        merged = merged.merge(smi, on="cid", how="inner")
        print(f"  {len(merged)} with SMILES")

    print("Standardising structures (this takes a minute for ~17k molecules)")
    std = [standardise(s) for s in merged["SMILES"]]
    merged["smiles_std"] = [a for a, _ in std]
    merged["inchikey"] = [b for _, b in std]
    merged = merged.dropna(subset=["smiles_std", "inchikey"])
    # `cid` is a string key ("cid:2519" / "sid:...") now, so it cannot be averaged
    merged = merged.groupby("inchikey", as_index=False).agg(
        {c: ("first" if c in ("smiles_std", "SMILES", "cid") else "mean")
         for c in merged.columns if c != "inchikey"})
    print(f"  {len(merged)} unique standardised structures")

    # ---- leakage guard, non-negotiable ---------------------------------
    test = pd.read_csv(args.test_csv)
    test.columns = [c.strip() for c in test.columns]
    tcol = next(c for c in test.columns if c.lower() == "smiles")
    tkeys = {k for _, k in (standardise(s) for s in test[tcol]) if k}
    hit = merged["inchikey"].isin(tkeys)
    print(f"\nLEAKAGE GUARD: {int(hit.sum())} external compound(s) match a BLIND TEST "
          f"compound -- removed.")
    merged = merged[~hit].reset_index(drop=True)

    train = pd.read_csv(args.train_csv)
    train.columns = [c.strip() for c in train.columns]
    trcol = next(c for c in train.columns if c.lower() == "smiles")
    trkeys = {k for _, k in (standardise(s) for s in train[trcol]) if k}
    overlap = merged["inchikey"].isin(trkeys)
    merged["in_challenge_train"] = overlap

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"\nwritten {args.out}")
    print(f"{'column':20s} {'labelled':>9s} {'NEW molecules':>14s}  {'note':>s}")
    for iso in ISOFORMS:
        pc, ac, cc = f"{iso}_pIC50", f"{iso}_active", f"{iso}_censored"
        if pc not in merged.columns:
            continue
        n = int(merged[pc].notna().sum())
        new = int((merged[pc].notna() & ~overlap).sum())
        ncen = int((merged.get(cc, 0) > 0).sum()) if cc in merged.columns else 0
        print(f"{pc:20s} {n:9d} {new:14d}  {n - ncen} fitted, {ncen} censored "
              f"at {CENSOR_PIC50}")
        if ac in merged.columns:
            n = int(merged[ac].notna().sum())
            new = int((merged[ac].notna() & ~overlap).sum())
            print(f"{ac:20s} {n:9d} {new:14d}  "
                  f"{100*merged[ac].mean(skipna=True):.1f}% active")
    print(f"\noverlap with the challenge training set: {int(overlap.sum())} "
          f"({100*overlap.mean():.1f}%)")
    print("The NEW-molecules column is what decides whether this is worth the GPU time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
