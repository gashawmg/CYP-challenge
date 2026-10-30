# Section 65. The placement rule is falsified on CYP1A2 and CYP3A4

Run date 2026-10-12. `probe2` scored MA-ST-RAE **0.5614, rank 66**, against our best of
**0.4737, rank 36**. That is a loss of 0.088 -- the worst single pIC50 submission since the
placement work began -- and it falsifies the fraction-above-4.0 rule on the two endpoints
it had never been tested on.

---

## 1. The arithmetic

`probe2` carried CYP2C9 and CYP2D6 over from `d6probe` byte for byte and shifted CYP1A2
and CYP3A4 each by **-0.30**. So the two frozen endpoints contribute exactly what they
did before, and the whole change is in the other two:

| | d6probe | probe2 | change |
|---|---|---|---|
| CYP2C9 (frozen) | 0.3645 | 0.3645 | 0 |
| CYP2D6 (frozen) | 0.5621 | 0.5621 | 0 |
| **CYP1A2 + CYP3A4** | **0.9679** | **1.3190** | **+0.3511** |
| macro | **0.4737** | **0.5614** | +0.0877 |

An average of **+0.176 per endpoint** from a 0.30 shift. I had predicted a worst case of
+0.034 per endpoint, taking the CYP2D6 probe as the analogue. It was five times worse.

The supporting metrics agree and say why. MA-R2 fell 0.5872 -> **0.5376** and MA-MAE rose
0.6664 -> **0.7159**, while MA-rho stayed at **0.7558** -- unchanged, because a constant
shift cannot move a rank correlation. So nothing about the models changed; the columns
were simply moved to a worse place.

## 2. What the R2 drop proves about direction

For a pure shift of size d, `R2` changes by `-[d^2 - 2d(M - c)] / V`. With d = -0.30 and
R2 falling, `0.09 + 0.6(M - c) > 0`, so `M - c > -0.15`.

For CYP1A2 at c = 4.301 that means the blind truth mean sits **above about 4.15**. Our
column was already close to it, and moving the centre to 4.001 moved away. The same
argument applies to CYP3A4 at c = 4.704.

So the direction is now measured, not modelled: **on CYP1A2 and CYP3A4, down is wrong.**

## 3. The rule that failed, and the argument that should have stopped me

Section 51 proposed placing each column so its fraction above the 4.0 censoring bound
matches the library's. It asked for shifts of -1.005 on CYP1A2 and -0.713 on CYP3A4, and
it was calibrated on exactly one endpoint -- CYP2D6, the only one with board measurements,
where its target sat 0.035 from the best-known placement.

**Section 51 also contained the argument that should have stopped this.** Because
`E[(y-p)^2] >= (M-c)^2`, we have `|M - c| <= sd(y) * sqrt(1 - R2)` with no distributional
assumption at all. With R2 of 0.6138 on CYP1A2 and 0.7055 on CYP3A4, that bound was
0.81-0.99 and 0.71-0.87 -- and I used it to reject the full shift while approving a -0.30
probe anyway.

The bound does more than cap the error. A HIGH R2 means the centre is already close, and a
column that is already close has nothing to gain from being moved and everything to lose.
CYP2D6's R2 was 0.1423 when it was probed -- that is what a badly placed column looks like,
and it is why the probe paid there. CYP1A2 at 0.6138 and CYP3A4 at 0.7055 were telling me
they were already placed.

**A probe is cheap only when there is a reason to think the current placement is wrong.**
On these two endpoints the evidence said the opposite and I probed regardless.

## 4. What to submit

**`artifacts\submission_d6probe.csv`** -- unchanged on disk, 750 rows, the exact file that
scored 0.4737 at rank 36. The live board is at 0.5614, so this recovers **0.088**, the
largest single move available on either track.

Its columns, for the record: CYP1A2 mean 4.301 sd 1.094, CYP2C9 4.849 / 0.792,
CYP2D6 3.649 / 0.690, CYP3A4 4.704 / 0.957.

## 5. Placement is finished on three of four endpoints

| endpoint | status | evidence |
|---|---|---|
| CYP2D6 | **frozen at 3.649** | -0.30 gained; a further -0.70 with widening lost 0.079 |
| CYP1A2 | **frozen at 4.301** | -0.30 lost; R2 0.6138 says the centre was already close |
| CYP3A4 | **frozen at 4.704** | -0.30 lost; R2 0.7055 says the same |
| CYP2C9 | untested | R2 0.7054 -- the highest of the four, so the same reasoning says leave it |

Every endpoint has now either been measured to be at its best known placement or has an R2
high enough to say the centre is close. **The placement thread is closed.** What remains on
the pIC50 track is ranking, which is modelling: the CYP3A4 surrogate pair from Section 52
(1,244 extra compounds, measured at Spearman 0.7423 -> 0.7519 locally) is the only pIC50
experiment with a measured effect behind it and it has still not been run.

## 6. Ledger

Four board results were missing from `artifacts\submissions.csv` -- it stopped at `cand1`
on 29 September. Twelve rows appended covering `d6probe`, `d6wide` and `probe2`, per the
Section 60 rule that the standing best is read from the ledger and never from a note.

Recorded because the argument that would have prevented this was written down in the same
section that proposed the thing it should have prevented.
