# Section 60. I was reading the wrong baseline. Both endpoints lost.

Run date 2026-10-12. Gashaw was right and Section 59 was wrong. The standing TDI position
was not MA-MCC 0.3325. It was **0.3694**, and `submission_tdi_structfree.csv` lost on
**both** endpoints, not one.

The correct record was in `artifacts/submissions.csv` the whole time -- the ledger that
`scripts/fetch_leaderboard.py` exists to maintain, whose own docstring says per-target
history "is what every placement and splice decision is made from, and transcription is
where those decisions silently go wrong." I made every decision in Sections 56 to 59 from
docstring prose instead, and the prose was three submissions stale.

---

## 1. The real standing position

From the ledger, TDI submissions in order:

| order | submission | CYP2D6 | CYP3A4 | MA-MCC | rank |
|---|---|---|---|---|---|
| 2 | splice | 0.1825 | 0.4470 | 0.3147 | 20 |
| 3 | tdi_w (fitted weights) | 0.2180 | 0.3798 | 0.2989 | 24 |
| 4 | tdi_best | 0.2180 | 0.4470 | 0.3325 | 15 |
| 6 | tdi_sel (greedy selection) | 0.2866 | 0.4035 | 0.3451 | 13 |
| 7 | tdi_v2 | 0.2918 | 0.3830 | 0.3374 | 15 |
| 8 | tdi_v3 | 0.2918 | 0.4074 | 0.3496 | 15 |
| **9** | **tdi_splice_best** | **0.2918** | **0.4470** | **0.3694** | **13** |
| -- | **structfree (today)** | **0.2727** | **0.4059** | **0.3393** | **32** |

**Standing best 0.3694. Today 0.3393. A loss of 0.0301, on both endpoints:**

| | standing | today | change |
|---|---|---|---|
| CYP2D6 | 0.2918 | 0.2727 | **-0.0191** |
| CYP3A4 | 0.4470 | 0.4059 | **-0.0411** |

## 2. What this does to Sections 58 and 59

Section 58's CYP2D6 ratio argument used **0.2180** as the baseline. That was `tdi_best`,
submission 4 of 9, superseded by `tdi_v2` three submissions later. The live CYP2D6 column
had been **0.2918** since 7 September.

So the claim "CYP2D6 beat its forecast, 0.2180 -> 0.2727" is wrong in the only way that
matters: the comparison was against a column we had already replaced. Measured against
what was actually live, the new CYP2D6 column **lost 0.019**.

Section 59's headline -- "not worse on the metric, the best this project has scored" -- is
withdrawn entirely. It was a regression of 0.030 and Gashaw's rank reading was the correct
signal.

Section 59's CYP3A4 analysis survives: the concentration pattern is real and now has a
fourth point. Ordered by how concentrated the column is:

| CYP3A4 column | blind MCC |
|---|---|
| equal-weight blend, ~20 members | **0.4470** |
| `tdi_v3`, equal-19 at rate 0.150 | 0.4074 |
| `lgb_clf` alone | 0.4059 |
| `tdi_sel`, greedy member selection | 0.4035 |
| fitted weights, `struct` at 0.29 | 0.3798 |

Nothing has ever beaten the broad equal-weight blend on CYP3A4. Greedy selection was tried
in September (`tdi_sel`) and scored 0.4035. Section 55 proposed greedy selection as new; it
is not new, and it has already been measured and lost.

## 3. `submission_tdi_bestmix.csv` is also wrong -- discard it

I built it last night as CYP3A4 0.4470 + CYP2D6 0.2727, predicting 0.3599. Against the
real ledger that is still **0.0095 below the standing best**, because 0.2727 is not the
best CYP2D6 column -- 0.2918 is.

The best combination of already-scored columns is CYP2D6 0.2918 + CYP3A4 0.4470 = 0.3694,
and that combination already exists: it is `submission_tdi_splice_best.csv`, the standing
best itself. There is no free recovery above it.

## 4. What to do

**Re-submit `artifacts/submission_tdi_splice_best.csv`** to restore 0.3694. It is on disk,
750 rows, CYP3A4 rate 0.2107 and CYP2D6 rate 0.0853, and it is the exact file that scored
0.3694 at rank 13. The live row today is 0.3393, so this recovers 0.030 and costs one slot.

Against the pIC50 slot: that board is at 0.4934 rank 47 against our own best of 0.4737, an
active loss of 0.0197. The TDI loss is 0.0301. TDI is the larger hole, but `probe2` also
buys two placement gradients where the TDI restore buys nothing but the restore. Either is
defensible; the TDI restore is the bigger single number.

## 5. The process failure, and the fix

Three sections of analysis rested on baselines read out of code comments written in early
September. `artifacts/submissions.csv` had the right numbers, indexed by submission and
endpoint, and I never opened it.

**Standing rule from here: every TDI and pIC50 claim about "the standing best" is read
from `artifacts/submissions.csv`, never from a docstring, a section note, or memory.** A
comment describes what was true when it was written; the ledger is maintained.

Also to fix: `scripts/run_tdi_ensemble.py` line 34 states "from submission_tdi_best, MA-MCC
0.3325, rank 15" and `scripts/tdi_splice.py` line 10 states "Predicted MA-MCC 0.3325;
actual 0.3325, rank 20 -> 15". Both are three and four submissions out of date and both are
load-bearing for anyone reading those files. They should say 0.3694.

Recorded because a user's one-line correction was worth more than three sections of my
own reasoning, and because the file that would have prevented it was written for exactly
this purpose.
