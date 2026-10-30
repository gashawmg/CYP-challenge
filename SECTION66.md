# Section 66. tdiF: the prior is dead, the alerts are within noise, and `pooled` moved

Run date 2026-10-13. `tdiF` tests two changes at once -- the ten FDA alerts added to
`tdi_alerts.py` (Section 64) and the two external-prior members (Section 63). Measured
against `tdiB` on identical folds at matched greedy depth:

| | tdiB | tdiF | difference | 95% CI | P(better) |
|---|---|---|---|---|---|
| CYP3A4 MCC @ 0.2915 | 0.4343 | 0.4437 | +0.0098 | [-0.0126, +0.0320] | 0.810 |
| CYP2D6 MCC @ 0.0706 | 0.1564 | 0.1774 | +0.0208 | [-0.0059, +0.0492] | 0.927 |
| macro | 0.2954 | **0.3106** | +0.0152 | | |

Both positive. Neither significant. Standing best is 0.3694, so nothing here ships.

My prediction before the run was "a few thousandths on CYP3A4 and nothing resolvable on
CYP2D6". Direction was right on significance, wrong on which endpoint moved: CYP2D6 moved
twice as much as CYP3A4, and both point estimates are larger than I said.

---

## 1. The external prior is dead

`lgb_prior` and `cat_prior` are **selected zero times out of five folds, on both
endpoints**, at greedy depth 12.

| member | CYP3A4 AUC | CYP2D6 AUC | times selected |
|---|---|---|---|
| lgb_prior | 0.8106 | 0.5859 | **0 / 5** |
| cat_prior | 0.7907 | 0.5700 | **0 / 5** |

`lgb_prior` is not a bad model -- fifth by AUC_macro in the member table, 0.8106 on
CYP3A4. It is a redundant one. Section 63 predicted it would "be a CYP3A4 member whose
value is diversity for the blend"; it has the quality and not the disagreement, so the
selector never takes it. That prediction is refuted.

Sixteen thousand external compounds, a feature block built in September and measured as
worth Spearman 0.495 -> 0.507 on the pIC50 track, and on TDI it buys nothing. The
mechanism is plausible -- a time-dependent inactivator has to bind first -- but whatever a
model learns from the qHTS panel about binding, `lgb_clf` has already learned from our own
3,584 rows.

## 2. The alerts in isolation moved +0.0015

`lgb_alert` uses only `cyp + tdialert`, so it is the clean readout of the alert block:

| | AUC_macro | MCC_CYP3A4 | MCC_CYP2D6 |
|---|---|---|---|
| lgb_alert, 51 alerts (tdiB) | 0.6611 | 0.3332 | 0.1179 |
| lgb_alert, 61 alerts (tdiF) | **0.6626** | 0.3402 | 0.1043 |

Ten new literature-derived columns moved the alert-only model by **0.0015 AUC**.

Yet the blend moved +0.015 macro. So whatever the new alerts contribute, they contribute
it through the larger models rather than through the alert-only one -- consistent with
Section 64's measurement that three of the ten clear 0.05 lift on CYP3A4 and one on
CYP2D6, which is a handful of extra splits for a tree that already has 2,200 columns.

## 3. The one member that clearly improved is `pooled`

| | AUC_macro | MCC_CYP3A4 | MCC_CYP2D6 | greedy picks, CYP3A4 |
|---|---|---|---|---|
| pooled (tdiB) | 0.6994 | 0.4052 | 0.1427 | 2 |
| **pooled (tdiF)** | **0.7006** | **0.4215** | **0.1641** | **3** |

`pooled` is the one model trained on all four isoforms at once with a one-hot isoform
indicator, and it is the member the new alerts helped most -- +0.0163 on CYP3A4 MCC and
+0.0214 on CYP2D6, moving it to third by AUC_macro and into the CYP3A4 selection.

That is a small thing but it points where Section 62 already pointed. The Novartis recipe
is multi-task regression on a continuous rate with the direct-inhibition pIC50s as
auxiliary tasks, and the only member here with a multi-task structure is the one that
responded to better mechanistic features. If the alerts help anything, they help the model
that is sharing a representation across endpoints.

## 4. CYP3A4 still picks `struct` first

Greedy picks, CYP3A4, depth 12: `struct` x7, `lgb_clf` x5, `svc_avalon` x5, `cat_screen`
x3, `pooled` x3, `lgb_screen` x3.

`struct` remains the top selection, as in every run since Section 54. Section 60's
constraint is unchanged and unaffected by anything in this section: the CYP3A4 submission
column stays frozen at the 0.4470 equal-weight blend, because `struct`-weighted CYP3A4 has
one board result and it is a 0.067 loss.

## 5. Where this leaves the TDI track

| | macro MCC |
|---|---|
| tdiB per-endpoint greedy | 0.2954 |
| **tdiF per-endpoint greedy** | **0.3106** |
| standing best, scored | **0.3694** |

Four structural changes have now been tried since Section 54: per-endpoint selection
(+0.023), Avalon (+0.023), five more fingerprint families (0.000), FDA alerts plus the
external prior (+0.015, not significant). None of it closes a 0.059 gap to a file that is
already on disk.

The remaining untried idea is the one from Section 62 that is a GPU evening rather than an
afternoon: **multi-task chemprop regression on the IC50 shift with the four direct-
inhibition pIC50s as auxiliary tasks, 10-model ensemble.** Section 66's one positive
result -- that the multi-task member is the one that responded -- is a small argument for
it. Everything else in the pool has been varied and has stopped moving.

The submission position is unchanged on both tracks:
`artifacts\submission_d6probe.csv` for pIC50 (0.4737, recovers 0.088 from the current
0.5614) and `artifacts\submission_tdi_splice_best.csv` for TDI (0.3694).

Recorded because two changes that each had a published paper behind them produced one
non-significant gain between them, and the only clear signal was in a member neither of
them was aimed at.
