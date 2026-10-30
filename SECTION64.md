# Section 64. The FDA alerts, measured before adding. Most of them do not transfer.

Run date 2026-10-12. Sixteen alerts were drafted from Faramarzi et al. (Front Pharmacol
2025) -- Table 4's qualified MBI alerts and Table 3's top CYP3A4 TDI features -- and every
one was measured on this library before any of it touched the production file. Ten went
in. **Six have essentially no coverage here, including the whole quinone-methide phenol
family that Section 62 called "the clearest hole".** That claim is withdrawn.

---

## 1. All sixteen, validated then measured

Every SMARTS compiles, matches its reference molecule, and none fires on aspirin. Then,
lift = P(TDI | alert fires) - P(TDI | it does not), on the labelled rows:

| alert | CYP3A4 n / lift | CYP2D6 n / lift |
|---|---|---|
| **fda_arene_phenethylamine** | 79 / **+0.144** | 90 / **-0.100** |
| **fda_alkoxy_alkylbenzene_ortho** | 125 / **+0.078** | 72 / +0.021 |
| **fda_alkylhalobenzene_ortho** | 110 / **+0.052** | 65 / -0.098 |
| fda_alkoxy_alkylbenzene_para | 160 / +0.032 | 57 / -0.006 |
| fda_amine_sec_cyclic | 62 / -0.004 | 78 / -0.080 |
| **fda_piperazine_tert** | 304 / -0.032 | 108 / **+0.056** |
| fda_alkylhalobenzene_para | 126 / -0.056 | 94 / -0.049 |
| fda_benzopyran | 31 / -0.085 | 18 / too few |
| fda_arene_alkene | 25 / -0.094 | 16 / too few |
| fda_diphenoxyethane | 40 / -0.114 | 7 / too few |
| fda_amine_primary_alkyl | **11** | 6 |
| fda_alkylphenol_para | **15** | 16 |
| fda_alkylphenol_ortho | **11** | 10 |
| fda_alkoxy_oxopropyl_benzene | **10** | 3 |
| fda_hydroquinone_para | **0** | 0 |
| fda_cyclopropyl_methyl_formamide | **0** | 0 |

## 2. The quinone-methide hole does not exist on this library

Section 62 argued the clearest gap was the quinone-METHIDE route -- para-alkylphenol,
para-hydroquinone, para-alkoxy alkylbenzene -- because the FDA ranks two of them in its
top five CYP3A4 TDI features.

Measured here: **para-hydroquinone matches 0 of 6,145 compounds. Alkylphenols match 10 to
16.** There is nothing to encode. The FDA's alerts were qualified on 620 approved drugs;
the OpenADMET library is different chemistry, and their most-cited motifs are simply not
in it.

The one quinone-methide member that does have coverage, para-alkoxy alkylbenzene (n 160),
returns +0.032 on CYP3A4 -- real but small, and smaller than the **ortho** form at +0.078,
which the FDA does not single out at all.

That is the general pattern across the table: where the FDA names a specific para
substitution, the ortho form transfers better here. Two for two.

## 3. The one genuinely interesting result: a sign flip

`fda_arene_phenethylamine` -- an arene with an aliphatic amine two carbons out -- is
**+0.144 on CYP3A4 and -0.100 on CYP2D6**, on 79 and 90 compounds. The same motif raises
the odds of TDI on one isoform and lowers them on the other.

A feature that flips sign between endpoints is worth more to a per-endpoint model than a
feature that is merely strong, because it cannot be imitated by anything the shared
descriptor block provides. Of all sixteen, this is the one I would watch.

`fda_piperazine_tert` does something similar at lower amplitude: -0.032 on CYP3A4,
**+0.056** on CYP2D6 with 304 and 108 hits. That is the FDA's own CYP2D6 story -- their
top five CYP2D6 features are all basic-amine motifs -- showing up in our data at the one
place it has coverage.

Six of the ten added have NEGATIVE lift on CYP3A4. That is not a reason to drop them: the
existing block's `acyl_glucuronide_acid` at -0.194 is one of its most useful columns,
because a compound headed for glucuronidation is not being bioactivated by a P450.

## 4. What changed on disk

`cypkit/tdi_alerts.py`: 51 columns -> **61**. Backup at `tdi_alerts.py.bak_pre_fda`.
All SMARTS compile, `calculate()` returns (n, 61) and still handles unparsable SMILES.
Each new entry carries its measured lift on both endpoints as a comment, so the next
reader does not re-derive it.

The six with no coverage were left out rather than added as dead columns.

## 5. To run

```
python scripts\run_tdi_ensemble.py --tag tdiF --save-oof
```

```
python scripts\tdi_per_endpoint_blend.py --tag tdiF
```

This also picks up `lgb_prior` and `cat_prior` from Section 63, so `tdiF` tests both
changes at once. If it matters later which one moved anything, the member table separates
them -- `lgb_alert` and `cat_screen` carry the alert block without the prior, and
`lgb_prior` carries the prior.

## 6. Expectations, stated before the run

Ten columns of which three have CYP3A4 lift above 0.05 and one has CYP2D6 lift above 0.05,
added to a block of 51 inside a model that already reaches AUC 0.8200 on CYP3A4. I expect
a few thousandths on CYP3A4 and nothing resolvable on CYP2D6. The sign-flipping
phenethylamine feature is the only one with a mechanism for doing better than that.

Nothing ships on it. `artifacts\submission_tdi_splice_best.csv` restores 0.3694, and the
CYP3A4 column stays frozen per Section 60.

Recorded because Section 62 named a hole from a paper and this section measured it to be
empty, which is the difference between reading a result and testing whether it applies to
your own compounds.
