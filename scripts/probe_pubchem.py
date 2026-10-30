#!/usr/bin/env python
"""Diagnostic: try every plausible route to the CYP panel and show what comes back.

    python scripts/probe_pubchem.py

Prints the HTTP status, content type, size and the first ~300 characters of the body
for each candidate URL. Nothing is cached, nothing is parsed, nothing is written --
this exists purely so the actual server responses can be read instead of guessed at.

Three of the earlier failures were mine: a wrong `aids_type` qualifier (HTTP 400), and
retrying deterministic 4xx errors. The 503s are different -- that is PubChem's
throttle ("PUGREST.ServerBusy"), so this probe waits a full second between calls and
sends a descriptive User-Agent, which is what NCBI asks for.

Paste the output back and the fetcher can be pointed at whichever route actually works.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
CGI = "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi"

CANDIDATES = [
    # --- does anything at all work? -------------------------------------
    ("sanity: caffeine SMILES",
     f"{REST}/compound/cid/2519/property/CanonicalSMILES/CSV"),
    ("sanity: AID 1851 summary",
     f"{REST}/assay/aid/1851/summary/JSON"),

    # --- routes to the child assays -------------------------------------
    ("child AIDs, plain",        f"{REST}/assay/aid/1851/aids/JSON"),
    ("child AIDs, TXT",          f"{REST}/assay/aid/1851/aids/TXT"),
    ("assay description",        f"{REST}/assay/aid/1851/description/JSON"),
    ("PUG-View assay record",    f"{VIEW}/data/assay/1851/JSON"),

    # --- routes straight to the DATA, skipping discovery ----------------
    ("PUG-REST assay CSV",       f"{REST}/assay/aid/1851/CSV"),
    ("classic datatable CGI",
     f"{CGI}?query=download&record_type=datatable&actvty=all&response_type=save&aid=1851"),
    ("concise datatable CGI",
     f"{CGI}?query=download&record_type=concise&actvty=all&response_type=save&aid=1851"),
]


def probe(name, url, timeout=90):
    print(f"\n{'='*72}\n{name}\n  {url}")
    req = urllib.request.Request(
        url, headers={"User-Agent": "cyp-challenge-research/1.0 (python-urllib)"})
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(400_000)
            print(f"  HTTP {r.status}  {r.headers.get('Content-Type','?')}  "
                  f"{len(body)} bytes read  ({time.time()-t0:.1f}s)")
        head = body[:300].decode("utf-8", errors="replace").replace("\n", " | ")
        print(f"  body: {head}")
        # if it is JSON, say what the top-level keys are
        try:
            j = json.loads(body)
            print(f"  JSON top-level keys: {list(j)[:8]}")
        except Exception:
            first = body.split(b"\n", 1)[0].decode("utf-8", errors="replace")
            if "," in first:
                cols = first.split(",")
                print(f"  CSV columns ({len(cols)}): {cols[:14]}")
        return True
    except urllib.error.HTTPError as e:
        detail = e.read(300).decode("utf-8", errors="replace").replace("\n", " | ")
        print(f"  HTTP {e.code} {e.reason}")
        print(f"  body: {detail}")
    except Exception as e:  # noqa: BLE001
        print(f"  {type(e).__name__}: {str(e)[:200]}")
    return False


def main():
    print("Probing PubChem. One second between calls to stay under the rate limit.")
    ok = []
    for name, url in CANDIDATES:
        if probe(name, url):
            ok.append(name)
        time.sleep(1.0)
    print(f"\n{'='*72}\nSUCCEEDED: {ok if ok else 'nothing'}")
    if not ok:
        print("If even the caffeine sanity check failed, something between you and\n"
              "PubChem is blocking it (proxy, VPN, firewall) rather than the URLs\n"
              "being wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
