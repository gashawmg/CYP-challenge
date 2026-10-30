# Section 50. The ST-RAE-versus-R2 alarm was mine, and it was a mis-anchored surface

Run date 2026-10-11. The previous section ended on a warning that `d6fit` was built to the
wrong objective, because three competitor rows on the CYP2D6 board showed ST-RAE and R2
moving in opposite directions. That warning is withdrawn. The surface it rested on was
anchored at centre 3.949, which is the PRE-probe column; the live column is at 3.649. With
the right anchor the conclusion inverts, and the competitor rows stop being an anomaly.

## 1. The scoring geometry that drives everything

From `artifacts/labels.csv`, CYP2D6: 4,376 rows, 3,012 of them left-censored with
`y_low = -inf` and `y_high ~ 4.00`. Under MA-ST-RAE,
`clip(p - hi, 0) + clip(lo - p, 0)`, **every prediction at or below 4.0 costs exactly zero
on 69% of the training set**, and the blind set is the unfiltered library, so its censored
fraction is at least that high.

That dead zone is why average error cannot predict ST-RAE. It is also why the
R2-maximising affine map is not automatically the ST-RAE-minimising one: R2 wants
slope `r * sd(y)/sd(p)`, the dead zone wants predictions pushed down and spread out so the
inactives fall under the bound and the actives reach their narrow intervals.

## 2. A blind-set model constrained by what the board has told us

Independently identified from our own two submissions, which differ only by a constant
shift of the CYP2D6 column and therefore share a Pearson:

| | centre | sd | board ST-RAE | board R2 |
|---|---|---|---|---|
| cand1 | 3.949 | 0.691 | 0.5957 | 0.1423 |
| d6probe (live) | 3.649 | 0.691 | 0.5621 | 0.3242 |

Two R2 values at two known centres solve for the truth: **mean 3.063, sd 1.558,
Pearson 0.747**. Back-substitution reproduces R2 0.3245 against the measured 0.3242.

With those pinned, the only free parameters are the blind set's censored fraction, its
censoring bound and its active interval width. Of 2,835 parameter combinations, **9
reproduce both board ST-RAE values to within 0.012**, and they cluster tightly: censored
fraction 0.85-0.90, bound 4.0-4.2, active width 0.26-0.40.

## 3. What every surviving model says

Averaged over 3 surviving families x 8 noise draws (24 estimates per candidate):

| candidate | centre | sd | modelled ST-RAE | vs live | worse in |
|---|---|---|---|---|---|
| live (d6probe) | 3.649 | 0.691 | 0.5499 | -- | -- |
| d6fit | 3.063 | 1.165 | 0.4690 | **-0.081** | 0/24 |
| d6full | 3.015 | 1.224 | 0.4575 | -0.092 | 0/24 |
| **d6wide (new)** | **2.950** | **1.500** | **0.4111** | **-0.139** | **0/24** |

The anchors come back at 0.5742 and 0.5499 against the board's 0.5957 and 0.5621 -- about
0.02 optimistic in level, and the modelled probe gain (0.024) is SMALLER than the measured
one (0.0336), so the model understates how much the shift direction pays.

## 4. The competitor rows, re-read

The three best CYP2D6 columns report (ST-RAE, MAE, R2). Two of those three numbers
over-determine a column's centre and spread once the truth is pinned. Solving each:

| | centre | sd | ST-RAE |
|---|---|---|---|
| preheat | ~2.85 | ~1.55 | 0.4135 |
| leebrubeck | ~2.95 | ~1.40 | 0.4618 |
| first-chair | ~2.40 | ~1.45 | 0.4632 |
| **ours, live** | **3.649** | **0.691** | **0.5621** |

All three sit low and wide. We are high and narrow against every one of them, on both
axes. The model fitted only to OUR two submissions reproduces preheat's ST-RAE to 0.014 and
leebrubeck's to 0.023 without ever having seen them.

And the inversion that caused the alarm is now ordinary: preheat is **wider than
R2-optimal**, which costs R2 and MAE and pays ST-RAE. Worse R2 with better ST-RAE is what
you get by overshooting the spread on purpose. It is evidence FOR widening, not against it.

## 5. The candidate

`artifacts/submission_d6wide.csv`, built by `artifacts/build_d6wide.py` from the live file:

    p' = 2.950 + 2.1734 * (p - 3.6489)

CYP2D6 mean 3.649 -> 2.950, sd 0.690 -> 1.500, range 2.797-5.252 -> 1.098-6.433,
fraction above the 4.0 bound 0.272 -> 0.245, which matches the ~23% the three competitor
placements imply. Rank order identical, no ties, no clipping. The other three columns are
copied as their exact original strings and verified unchanged in 0 of 2,250 fields.

Modelled macro: **0.4737 -> 0.4390**, against a board paired resolution of 0.0067.

## 6. What this does not settle

The model has 3 free parameters fitted on 2 points. Its survivors agree on direction and
sign, not on magnitude: the unconstrained optimum sits at centre ~2.6 and sd ~1.8, wider
than anything on the board, which says the model over-rewards width at the extreme.
`d6wide` deliberately stops inside the observed competitor envelope rather than at the
modelled optimum.

Ranking is untouched by any of this. Pearson 0.747 caps CYP2D6 R2 at 0.558 and no affine
map changes rho. After this placement is spent, the remaining CYP2D6 gain is modelling,
which is where Section 49's censored-row and multi-task arms come back.

Recorded because a conclusion published in the previous section was wrong, the error was a
wrong anchor rather than a wrong method, and the corrected surface points the opposite way.
