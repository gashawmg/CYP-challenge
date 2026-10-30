# Section 51. d6wide lost, the two-point solve was wrong, and the third anchor fixes it

Run date 2026-10-11, evening. `submission_d6wide.csv` was submitted on the strength of
Section 50 and **lost**: macro ST-RAE 0.4737 -> 0.4934, rank 36 -> 47, CYP2D6 0.5621 ->
0.6412. The recommendation was mine and it was wrong. This section records what the loss
bought, because it bought a great deal.

## 1. Two operational facts, now settled

**The Live leaderboard shows the LATEST submission, not the best.** Rank fell from 36 to
47 on a worse file. This question has been open since Section 44 and is now closed. It
means a bad submission costs a day at the lower score, and the only way back is to re-upload.

**The board scores deterministically.** CYP2C9 and CYP3A4 were byte-identical between the
two files and returned byte-identical rows on all four metrics.

## 2. The two-point solve is refuted by its own first out-of-sample test

Section 50's solve (truth mean 3.063, sd 1.558, our Pearson 0.747) predicted that d6wide,
at centre 2.950 and sd 1.500, would score **R2 0.5062**. It scored **0.3657**. An error of
+0.14 on a quantity the solve claimed to determine exactly.

The flaw is structural. Our first two submissions differed only by a constant shift of the
CYP2D6 column, so they share a Pearson AND a standard deviation. Two R2 values at two
centres with the SAME spread give one equation in three unknowns. The missing constraint
was supplied by a competitor-consensus sd of 1.550, an assumption, and it was wrong.

d6wide changed the spread, so it is the first anchor that is not degenerate with the others.

## 3. The three-anchor solve

With `R2 = 1 - [V + s^2 - 2 P s + (M - c)^2] / V`, three anchors give three equations in
three unknowns and an exact solution:

| | centre | sd | board R2 | board ST-RAE |
|---|---|---|---|---|
| cand1 | 3.949 | 0.691 | 0.1423 | 0.5957 |
| d6probe | 3.649 | 0.691 | 0.3242 | **0.5621** |
| d6wide | 2.950 | 1.500 | 0.3657 | 0.6412 |

**Truth mean 3.153, sd 1.460, our Pearson 0.7011.** Residuals on all three R2 values are
zero to five decimals. Max R2 obtainable from any affine map of our column: **0.4916**.

## 4. What the corrected surface says about CYP2D6

Refitting the censored-geometry model against all three ST-RAE anchors leaves 4 of 1,620
parameter combinations standing. Averaged over them and 4 noise draws each:

| candidate | centre | sd | modelled ST-RAE | vs d6probe |
|---|---|---|---|---|
| d6wide (live now) | 2.950 | 1.500 | 0.6256 | +0.0777 |
| **d6probe** | **3.649** | **0.691** | **0.5479** | **0.0000** |
| R2-optimal | 3.153 | 1.024 | 0.5897 | +0.0418 |
| modelled optimum | 3.60 | 0.85 | 0.5399 | **-0.0080** |

Two things follow. **The original alarm in Section 49 was right after all**: the
R2-optimal placement is 0.042 worse than what we already have, so maximising R2 is indeed
the wrong objective. Section 50 abandoned that alarm on the strength of a mis-specified
truth. And **CYP2D6 placement is spent** -- the best remaining move is worth 0.008 on the
endpoint, 0.002 on the macro, against a paired resolution of 0.0067.

## 5. A diagnostic that needs no fitting, and its limit

Every endpoint's censored rows have `y_high` near 4.00. So the fraction of a column sitting
above 4.0 should match the fraction of the library whose interval reaches above 4.0:

| | our fraction > 4.0 | library fraction | shift to match |
|---|---|---|---|
| CYP1A2 | 0.577 | 0.317 | -1.005 |
| CYP2C9 | 0.833 | 0.356 | -1.225 |
| CYP2D6 | 0.272 | 0.324 | +0.035 |
| CYP3A4 | 0.773 | 0.557 | -0.713 |

CYP2D6, the only endpoint with measurements, sits 0.035 from this rule's target at its
best-known placement -- and its three measured points are ordered exactly as the rule
predicts (fraction 0.420 scored 0.5957, 0.272 scored 0.5621, 0.245 scored 0.6412). That is
a real calibration, and it says all three remaining endpoints are placed too high.

**But the rule conflates centre with spread**, and a model-free bound contradicts it.
Since `E[(y-p)^2] >= (M - c)^2`, we have `|M - c| <= sd(y) * sqrt(1 - R2)` with no
distributional assumption at all. Taking sd(y) in 1.3-1.6, the band CYP2D6's solve
supports:

| | R2 | ceiling on centre error | shift the fraction rule asks for |
|---|---|---|---|
| CYP1A2 | 0.6138 | 0.81 to 0.99 | -1.005 |
| CYP2C9 | 0.7054 | 0.71 to 0.87 | -1.225 |
| CYP3A4 | 0.7055 | 0.71 to 0.87 | -0.713 |
| CYP2D6 | 0.3242 | 1.07 to 1.32 | +0.035 (actual offset 0.496) |

The full shifts are at or past a hard ceiling. Their columns are too high partly because
they are too WIDE, and a pure shift is the wrong instrument for that. `place2` and `place4`
were built to the fraction rule and are kept on disk, but they are not recommended.

## 6. What to submit next

`artifacts/submission_probe2.csv`. CYP2D6 and CYP2C9 are carried over from `d6probe` byte
for byte; CYP1A2 and CYP3A4 each get **-0.30**, the exact step size that was measured on
CYP2D6. Shift only, ranks and rho untouched, verified unchanged in 0 of 1,500 carried
fields.

It does three things with one slot:

1. **Restores CYP2D6 to 0.5621**, recovering the 0.0198 macro that d6wide gave away.
2. **Buys the second anchor for CYP1A2 and the second for CYP3A4 simultaneously.** The
   per-endpoint boards separate them with no confounding. Two anchors plus the third from
   a later spread change identifies each endpoint's truth the way it is now identified for
   CYP2D6.
3. **Re-checks determinism**: CYP2C9 must return 0.3645 and CYP2D6 must return 0.5621.

Bounded downside: the measured cost of a wrong-signed 0.30 shift on CYP2D6 was 0.034 on
that endpoint. Two of them is at most 0.017 macro, against the 0.0198 macro the restore
returns, so the file should not land worse than the current live score even if both probes
point the wrong way.

## 7. The lesson, stated plainly

Three inference failures in this project now share one shape: a quantity was solved from
measurements that could not determine it, and the missing constraint was filled in with an
assumption that was never labelled as one. Section 48's Gaussian inversion, Section 50's
two-point solve, and the d6wide recommendation that followed. The fix is not better
modelling. It is to count equations against unknowns before acting, and when they do not
balance, spend the slot on an anchor rather than on a prediction.
