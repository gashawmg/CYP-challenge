# Section 78 -- Cost-weighted training works. It is too small to be worth a campaign.

    python scripts\cost_weighted_probe.py --seeds 0 1 2    (LightGBM/ECFP4, frozen folds)

CONFIRMED AT THREE SEEDS. The single-seed run gave +0.71%; averaging three seeds gives
**+0.56%**. Predicted beforehand at ~+0.58%, so the shrinkage behaved as expected: part of
the single-seed gap was the particular seed drawn for the weighted arm, and averaging
removes that luck from both arms. The UNIFORM arm gains more from averaging (it had more
room: CYP1A2 placed 0.8503 -> 0.8437), which is why the gap narrows by about a fifth.
The three-seed table is the one to cite.

THREE SEEDS (authoritative):

| power | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | macro placed | vs uniform |
|---|---|---|---|---|---|---|
| 0.0 (uniform) | 0.8437 | 0.8474 | 0.9298 | 0.6579 | 0.8197 | baseline |
| 0.5 | 0.8374 | 0.8463 | 0.9245 | 0.6574 | 0.8164 | +0.40% |
| **1.0** | **0.8365** | 0.8453 | **0.9237** | **0.6549** | **0.8151** | **+0.56%** |
| 1.5 | 0.8401 | 0.8442 | 0.9261 | 0.6570 | 0.8168 | +0.35% |
| 2.0 | 0.8409 | **0.8430** | 0.9270 | 0.6592 | 0.8175 | +0.27% |

per-endpoint deltas against uniform, three seeds:

| power | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|
| 0.5 | +0.0064 | +0.0011 | +0.0053 | +0.0005 |
| 1.0 | +0.0072 | +0.0021 | +0.0061 | +0.0030 |
| 1.5 | +0.0036 | +0.0032 | +0.0037 | +0.0009 |
| 2.0 | +0.0029 | **+0.0044** | +0.0028 | -0.0013 |

## 1. The evidence is better than the effect size

**15 of 16 endpoint-power cells positive** across the whole three-seed sweep. One-sided
sign test p = 0.00026. The dose-response is cleanly unimodal on three endpoints -- noise
does not produce a smooth rise-then-fall over four settings on several independent
endpoints at once. Plain rho improves too (CYP1A2 0.4392 -> 0.4500), so this is not purely
an allocation trade; the weighting makes the model rank better as well as spend better.

**CYP2C9 PEAKS AT 2.0 -- CHECKED, AND IT IS A CEILING, NOT A RAMP.** Its delta rises
monotonically through the main sweep (+0.0011, +0.0021, +0.0032, +0.0044), which looked
like an optimum beyond the range tested. Extending it says otherwise:

    power  2.0  placed 0.8430      <- the optimum
    power  3.0         0.8535  (-0.0105)
    power  4.0         0.8553  (-0.0123)
    power  6.0         0.8636  (-0.0206)

So CYP2C9's best is about 2.0 and its maximum gain is the +0.0044 already measured. Using
per-endpoint optima (1.0 / 2.0 / 1.0 / 1.0) instead of a global 1.0 gives macro +0.0052
against +0.0046 -- **a further +0.0006, or 0.07%**, before paying for the selection bias of
choosing four powers on the rows that score them. A single global power of 1.0 is the right
setting and per-endpoint tuning is not worth the exposure.

**READ `raw ST-RAE` AGAINST `PLACED` AS A LESSON IN WHY THE YARDSTICK CHANGED.** On CYP1A2
at power 2.0 the RAW score improves 0.9339 -> 0.8616, a 7.7% gain that would look like a
major result. The PLACED score improves only 0.8437 -> 0.8409, 0.3%. Almost all of the raw
gain is placement, which a fitted affine already collects. Section 69 established that;
this run is a clean demonstration of it.

The Section 69 correction is therefore confirmed in the direction that matters: weighting
by the MARGINAL charge `P(|e| > h)` helps, where 1/width was rejected for the wrong reason.
Both arms of the U needed to be in the weight.

## 2. Why it still does not justify the GPU time

Section 75 measured that **12% of a member-level gain survives the four-member blend**.

    one member cost-weighted   0.56% x 0.12  =  0.07%   ->  0.0003 on the board, nothing
    ALL FOUR members           ~0.56%                   ->  0.0027 on the board, one rank

The four members are four separate codebases (v6, cp, unimol, the multihead). Retraining
all of them buys about one rank. That is a campaign priced far above what it returns, and
the standing rule against small modifications applies to effort as much as to slots.

**What to do with it instead:** keep the finding, and apply cost weighting whenever a
member is being retrained anyway for another reason. It is free at that point. Do not
start a retraining campaign to collect it.

## 3. Scope, stated where it was measured

LightGBM on ECFP4 at macro rho 0.4305, well below the production multihead's 0.6516.
Whether 0.71% holds at the top of the model range is untested, and the one-seed caveat the
script prints still stands -- `--seeds 0 1 2` halves the noise floor for minutes of CPU and
should be run before this section is cited as settled.

## 4. How it compares to the move actually on the table

| lever | cost | expected board effect | diluted by the blend? |
|---|---|---|---|
| cost-weighted training, one member | GPU hours | +0.0004 | yes, 12% survives |
| cost-weighted training, all four | GPU days | +0.0034 | no |
| **CYP1A2 +0.20 placement probe** | **one slot** | **unknown, plausibly 10x larger** | **no -- placement passes through whole** |

Placement changes are not diluted: they act on the submitted column directly. That, and not
the effect size, is why the probe outranks the training work for the next slot.
