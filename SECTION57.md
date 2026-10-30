# Section 57. The two runs did not do what I prescribed, and struct is back in both files

Run date 2026-10-12. The commands in Section 56 were incomplete. `--learners` restricts the
learner POOL, but `struct` and `pooled` are added by separate switches -- `--no-struct` and
`--no-pooled` -- which I did not pass. So both runs trained the members I named **plus
`struct` and `pooled`**, and both submission files were written from the equal-weight rank
blend of all of them.

The CYP3A4 column of `submission_tdi3a4.csv` therefore carries `struct` at weight **1/3**,
which is more than the 0.29 it carried in the fitted blend that took CYP3A4 from 0.4470 to
0.3798 on the board.

---

## 1. Neither file is submittable as written

Each run writes a full two-endpoint file, so either could be uploaded whole. Neither
should be. From the runs' own tables:

| file | CYP3A4 MCC | CYP2D6 MCC |
|---|---|---|
| `submission_tdi3a4.csv` | 0.4489 | **0.1527** |
| `submission_tdi2d6.csv` | **0.3979** | 0.2069 |

Each is good on the endpoint it was built for and bad on the other, as designed. They have
to be spliced, and `scripts/tdi_splice.py` exists for that.

Note also that those MCCs are at each run's **MCC-optimal** threshold, not at the blind
rate. They are not comparable to any number in Sections 54 to 56. Everything below is
recomputed at the blind rate with the cut taken out of fold, on the tdiB member OOF.

## 2. Like-for-like, and struct is the whole story again

**CYP3A4, blind rate 0.2915:**

| members | AUC | MCC |
|---|---|---|
| `lgb_clf` alone | 0.8200 | **0.4166** |
| `lgb_clf` + `pooled` | 0.8223 | 0.4153 |
| `lgb_clf` + `struct` + `pooled` -- what tdi3a4 shipped | 0.8346 | 0.4280 |
| `lgb_clf` + `struct` | 0.8340 | **0.4355** |

The shipped column beats the single model by +0.0111, 95% CI [-0.0115, +0.0344]. Drop
`struct` and the blend falls to 0.4153, below the single model. **Every point of CYP3A4
blending gain is `struct`, and the more weight it carries the better it looks locally** --
0.4280 at one third, 0.4355 at one half. That is the third independent time this endpoint
has said the same thing, and the one time it was acted on it cost 0.067 on the board.

**CYP2D6, blind rate 0.0706:**

| members | AUC | MCC |
|---|---|---|
| `knn_fcfp` alone | 0.6283 | 0.1501 |
| 6 similarity only | 0.6488 | 0.1669 |
| 6 similarity + `struct` + `pooled` -- what tdi2d6 shipped | 0.6448 | 0.1774 |
| **6 similarity + `pooled`, no struct** | 0.6474 | **0.1796** |

Here the accident was useful: `pooled` adds +0.013 over similarity alone, which Section 56
would have excluded. `struct` costs 0.002. So the best CYP2D6 configuration is the six
similarity members plus `pooled`, with `struct` dropped -- and unlike CYP3A4, nothing in
it depends on `struct`.

## 3. Corrected commands

```
python scripts\run_tdi_ensemble.py --tag tdi3a4b --learners lgb_clf --no-struct --no-pooled --predict
```

```
python scripts\run_tdi_ensemble.py --tag tdi2d6b --learners knn_fcfp svc_avalon1k svc_c10 knn_tt svc_tan svc_avalon --no-struct --predict
```

The second keeps `pooled` deliberately and drops only `struct`. Then splice CYP3A4 from
the first and CYP2D6 from the second with `scripts\tdi_splice.py`.

## 4. The honest arithmetic, and why this still does not ship

Macro MCC at blind rates, cuts out of fold:

| configuration | CYP3A4 | CYP2D6 | macro |
|---|---|---|---|
| struct-free, as corrected above | 0.4166 | 0.1796 | **0.2981** |
| what the two runs actually produced, spliced | 0.4280 | 0.1774 | 0.3027 |
| per-endpoint greedy (Section 55) | 0.4379 | 0.1881 | 0.3130 |
| **standing best, actually scored** | | | **0.3325** |

**No configuration of this pool clears the standing best locally**, and the three that come
closest all do it by weighting `struct` on CYP3A4. The struct-free choice costs about 0.005
of local macro and removes the one failure mode this project has measured on the board.

The single local-to-board anchor (OOF 0.2524 scoring 0.285, +0.033) would put 0.2981 at
roughly 0.331 and 0.3027 at 0.336 -- brushing the standing best from either side. That is
not a margin worth a slot on a one-point extrapolation, which this project has got wrong
in Sections 48, 50 and 53.

So the TDI track stands where Section 55 left it, with one thing gained: the CYP2D6
configuration is now better specified (six similarity members plus `pooled`, 0.1796 against
the 0.1361 the 20-member equal blend gave) and the CYP3A4 question is reduced to a single
yes-or-no that only the board can answer -- is `struct` safe at weight on CYP3A4, given it
is the only thing that ever improves that endpoint locally and the only thing that has ever
damaged it live.

Recorded because an incomplete command put the suspect member back into both candidate
files at a higher weight than the one that caused the original damage.
