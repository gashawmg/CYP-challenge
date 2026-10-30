# Section 67. The best TDI submission is 0.376, and the best CYP3A4 column is 0.4601

Run date 2026-10-13. Gashaw was right again. The standing TDI best is **`tdi_blendcp` at
MA-MCC 0.376, rank 9**, not `tdi_splice_best` at 0.3694. More importantly the best CYP3A4
column ever scored is **0.4601 at rank 8**, not 0.4470 -- and that number has been
load-bearing for Sections 59, 60, 62 and 66.

The ledger stopped at order 9 on 8 September. Three TDI submissions followed on 12-14
September and none was recorded. The scores were sitting in the docstrings of the scripts
that built them.

---

## 1. The corrected record

| ord | submission | CYP3A4 | CYP2D6 | MA-MCC | rank |
|---|---|---|---|---|---|
| 9 | tdi_splice_best | 0.4470 | 0.2918 | 0.3694 | 13 |
| 10 | tdi_blend | -- | -- | not recorded anywhere | -- |
| **11** | **tdi_blendcp** | **0.4601** (rank 8) | **0.2918** | **0.376** | **9** |
| 12 | tdi_blend2d6 | 0.4601 (copied) | changed, lost | 0.3312 | 32 |
| 13 | tdi_structfree | 0.4059 | 0.2727 | 0.3393 | 32 |

Verified on disk: `blend2d6`'s CYP3A4 column is identical to `blendcp`'s on 750/750 and its
CYP2D6 differs on 110 compounds, which is exactly what its own docstring says it did. Call
counts are 158/750 and 64/750 across all three. The 0.3312 for `blend2d6` is taken from
Section 42's standing-position line rather than a direct attribution, so it carries that
caveat in the ledger.

Eight rows appended to `artifacts\submissions.csv`.

## 2. What this does to the CYP3A4 story

Sections 59 and 60 built a case that CYP3A4's blind set rewards breadth, ranking the
columns by how concentrated they are. The table was right in shape and wrong at the top:

| CYP3A4 column | blind MCC |
|---|---|
| **equal-weight blend + chemprop shift model (`blendcp`)** | **0.4601** |
| equal-weight blend, ~20 members (`tdi_best`, `splice_best`) | 0.4470 |
| tdi_v3, equal-19 | 0.4074 |
| `lgb_clf` alone | 0.4059 |
| tdi_sel, greedy member selection | 0.4035 |
| fitted weights concentrated on `struct` | 0.3798 |

The breadth conclusion survives -- every concentration still loses, in the same order. But
the best column is not the plain blend. **It is the blend plus one decorrelated member,
and that member is a ChemProp model trained on the continuous IC50 shift.**

`tdi_blend_cp.py` records what it bought and how: d' 0.904 -> 1.008, +0.103 with a 95%
interval of [+0.055, +0.154] and P(worse) 0%, holding at +0.096 [+0.073, +0.120] under a
split-half selection test. It scores WORSE standalone (d' 0.889 against 0.904) and its rank
correlation with the incumbent is **0.644**. It paid purely by disagreeing -- the same
mechanism as Avalon on CYP2D6, and the only mechanism that has ever produced a real gain
on either track.

## 3. This also corrects Section 62

Section 62 called the Novartis multi-task shift regression "the biggest single untried
idea". A single-task version of it is already in our best submission and already paid
+0.0131 on CYP3A4 at rank 8. The untried part is narrower than I said: multi-task with
the four direct-inhibition pIC50s as auxiliary tasks, and a 10-model ensemble. The target
reformulation itself is proven.

That is a better-founded experiment than I made it sound, not a worse one -- it is an
extension of the one thing on this track with a board-confirmed, bootstrap-clean gain.

## 4. Every "standing best" figure since Section 56 was 0.0066 low

Sections 56, 57, 58, 59, 60, 63 and 66 all compared candidates against 0.3694. The real bar
is 0.376. None of those conclusions flips -- the best thing this pool produced was 0.3130
local, nowhere near either number -- but the arithmetic in each should be read with the
corrected bar, and Section 60's "there is no free recovery above 0.3694" was wrong: there
was, and it is `blendcp`.

## 5. What to submit

**`artifacts\submission_tdi_blendcp.csv`** -- on disk, 750 rows, CYP3A4 158/750 calls and
CYP2D6 64/750, the exact file that scored 0.376 at rank 9. The live TDI row is `structfree`
at 0.3393, so this recovers **0.037**.

The best combination of separately scored columns is CYP3A4 0.4601 and CYP2D6 0.2918, and
`blendcp` already is that combination. There is nothing better to assemble from what has
been measured.

## 6. The process failure, twice in two days

Section 60 established the rule -- read the standing best from `artifacts\submissions.csv`,
never from a note -- after the same class of error on the TDI baseline. I then followed it
to a ledger that was itself three submissions stale, and did not check whether the ledger
ended before the last submission.

The amended rule: **read the ledger, then check its last entry against the newest
submission file on disk.** `artifacts\submission_tdi_blendcp.csv` is dated 14 September;
the ledger stopped on 8 September. Six days and three submissions of drift were visible
from a directory listing.

I have appended the missing TDI rows and, in Section 65, the missing pIC50 rows. Both
tracks now run to the present.

Recorded because the correct numbers were in the docstrings of our own scripts, and I had
read two of those scripts in this session without noticing the board results in them.
