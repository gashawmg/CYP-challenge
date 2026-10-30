
## pIC50  (MA-ST-RAE, lower is better)

| submission | MA-ST-RAE | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | what changed |
|---|---|---|---|---|---|---|
| l1 | 1.0127 | — | — | — | — | classical L1 baseline |
| dense_blend | 0.9171  ↑0.0956 | — | — | — | — | dense-fitted blend |
| composite_scaled | 0.8244  ↑0.0927 | — | — | — | — | composite blending |
| all7_calib | 0.6763 (r49)  ↑0.1481 | — | — | — | — | 7 members + calibration |
| v3_blend | 0.7056 (r51)  ↓0.0293 | — | — | — | — | + ChemProp v3 (censored training) |
| v3_affine80 | 1.1048 (r118)  ↓0.3992 | — | — | — | — | wrong-sign affine correction |
| pseudo_blend45 | 0.7273 (r58)  ↑0.3775 | 0.5791 (r37) | 0.4627 (r39) | 1.3092 (r65) | 0.5581 (r57) | + screen pseudo-labels; reblended; raw |
| placed | 0.5837 (r34)  ↑0.1436 | 0.5495 (r32) | 0.5055 (r49) | 0.7034 (r40) | 0.5763 (r67) | placement optimisation |
| blend3_placed | 0.5398 (r29)  ↑0.0439 | 0.5153 (r27) | 0.4547 (r39) | 0.6597 (r33) | 0.5295 (r47) | 3-way blend + placement |
| blend3_narrow | 0.5454 (r32)  ↓0.0056 | 0.6300 (r54) | 0.3978 (r17) | 0.6602 (r36) | 0.4938 (r40) | width-only re-placement; CYP1A2 mean also moved |
| **splice_best** | **0.5167 (r28)  ↑0.0287** | 0.5153 (r24) | 0.3978 (r17) | 0.6602 (r36) | 0.4938 (r40) | per-endpoint splice of scored columns |

  best overall: splice_best at 0.5167
  best ever per target: CYP1A2 0.5153 (blend3_placed) · CYP2C9 0.3978 (blend3_narrow) · CYP2D6 0.6597 (blend3_placed) · CYP3A4 0.4938 (blend3_narrow)
  splicing those gives 0.5167  ->  +0.0000 better than the best single submission, with no modelling

## TDI  (MA-MCC, higher is better)

| submission | MA-MCC | CYP3A4 | CYP2D6 | what changed |
|---|---|---|---|---|
| mtl_blend | 0.2981 | — | — | multitask blend |
| splice | 0.3147 (r20)  ↑0.0166 | 0.4470 (r4) | 0.1825 (r27) | per-endpoint splice |
| tdi_w | 0.2989 (r24)  ↓0.0158 | 0.3798 (r33) | 0.2180 (r20) | fitted blend weights on CYP3A4 |
| tdi_best | 0.3325 (r15)  ↑0.0336 | 0.4470 (r4) | 0.2180 (r20) | best column per endpoint |
| tdi_recut | 0.3187 (r18)  ↓0.0138 | — | — | new score column + re-cut |
| **tdi_sel** | **0.3451 (r13)  ↑0.0264** | 0.4035 (r25) | 0.2866 (r13) | greedy member selection |
| tdi_v2 | 0.3374 (r15)  ↓0.0077 | 0.3830 (r38) | 0.2918 (r13) | best column per endpoint + re-cut at the equal-variance optimum |

  best overall: tdi_sel at 0.3451
  best ever per target: CYP3A4 0.4470 (splice) · CYP2D6 0.2918 (tdi_v2)
  splicing those gives 0.3694  ->  +0.0243 better than the best single submission, with no modelling

ledger: /home/claude/cyp/artifacts/submissions.csv
