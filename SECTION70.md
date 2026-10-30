# Section 70 -- The surrogate worked, which says the TDI arm belongs as a HEAD, not as rows

## 1. The CYP3A4 surrogate pair, finally run

`base3a4` against `surr3a4`, identical folds, identical seed, the only difference being
1,238 surrogate rows at weight 0.50:

| | rho | bottom 10% | middle 80% | top 10% | AUC hi vs lo | raw ST-RAE | **placed ST-RAE** |
|---|---|---|---|---|---|---|---|
| base3a4 | 0.7613 | 0.0575 | 0.6695 | 0.1809 | 0.9883 | 0.5439 | **0.4991** |
| surr3a4 | 0.7748 | 0.0446 | 0.6819 | 0.2082 | 0.9917 | 0.5295 | **0.4837** |

Paired bootstrap over 2,000 resamples of the 2,335 compounds, both arms placed optimally
first so the comparison is ranking against ranking:

    d rho             +0.0135   95% CI [+0.0022, +0.0250]   P(worse)  0.9%
    d placed ST-RAE   +0.0151   95% CI [-0.0011, +0.0314]   P(worse)  3.8%

The rho interval excludes zero. The ST-RAE interval grazes it. The two arms' rankings
correlate 0.9164, so this is a real effect on a largely shared column, not two different
models. **The TDI arm carries CYP3A4 signal the scored column does not have.**

Where it landed: middle-80% rho +0.0124, top-decile rho +0.0273, bottom decile -0.0129.
Raw cost fell most in the top decile (144.7 -> 131.6, -9%).

## 2. Why this result argues AGAINST doing more of it

The surrogate route converts a TDI-condition measurement into a pseudo-pIC50
(`y = 0.9148*T + 0.0391`) and pours it into the scored column. That is precisely the move
`cypkit/multihead.py` was written to stop doing:

> Every extra source measures something adjacent to the scored target, not identical to
> it. Pooling any of them into the scored columns needs a cross-assay correction; as
> separate heads none is required.

The surrogate needed exactly such a correction, we fitted it by regression, and it still
gained. That is evidence the underlying signal is strong enough to survive a lossy
conversion -- not evidence the conversion is a good idea. `train_cyp_mh.py` has no
`--extra_csv` and should not get one.

## 3. What the challenge actually hands us, unused

`data/cyp-challenge-TRAIN_TDI.csv` has 6,145 compounds and four full dose-response
columns, `<iso>_pIC50_TDI_condition`, with their own confidence bounds. The multihead
table has never touched them:

| endpoint | TDI-arm curves | scored curves | TDI compounds with NO scored curve |
|---|---|---|---|
| CYP1A2 | 1,413 | 1,412 | 1 |
| CYP2C9 | 1,285 | 1,285 | 0 |
| CYP2D6 | 1,497 | 1,493 | 4 |
| **CYP3A4** | **3,583** | **2,335** | **1,249** |

1,240 of the 6,145 TDI compounds are absent from the inhibition file entirely, and **zero
are in the blind set** -- checked directly, not assumed. So four new heads buy 1,249
genuinely new CYP3A4 compounds plus a paired second reading on roughly 1,400 compounds per
endpoint. Same assay, same deck, different condition: the exact shape the architecture
wants, with no harmonisation.

## 4. The leakage argument, which is the only thing that could sink this

A compound's TDI-condition curve is highly correlated with its scored direct-inhibition
curve. Handing the model one while scoring the other would be a read-off, and would show
up as an inflated OOF that dies on the board -- the failure this project has least
appetite for.

It is safe here for one specific reason, and only that reason: `train_cyp_mh.py` holds out
whole ROWS, not individual cells.

    va = fold == k
    tr = ~va                      # external (fold -1) is always in training

A validation compound's TDI head is withheld along with its scored head. The same
exposure already exists for the `ch_<iso>_lfc` screen heads and is handled the same way.
The 1,249 CYP3A4-only compounds get fold -1, live permanently in training, and carry no
scored label, so they can never be read off. A comment to this effect is now in
`multihead.py` beside the loading block: **if anyone ever changes that masking to
cell-level, these heads must come out first.**

## 5. The change

`cypkit/multihead.py` gains four `ch_<iso>_tdi` heads behind a `with_tdi` flag, and
`train_cyp_mh.py` gains `--no-tdi` for the ablation. Backups at `multihead.py.bak_pre_tdi`
and `train_cyp_mh.py.bak_pre_tdi`. Nothing is converted and no existing head moves.

## 6. How it will be judged

Matched pair, one flag apart, scored with `scripts/rho_yardstick.py`, which places both
arms optimally before comparing. The existing `oof_st_mh_ctl.csv` is NOT used as the
control: its settings cannot be recovered from disk, and a stale baseline has already cost
this project three wrong calls (Sections 51, 67, 68). Both arms get run.

Success is a rho gain whose paired-bootstrap interval excludes zero, on more than one
endpoint. A CYP3A4-only gain would be the surrogate result again by a cleaner route and
would not on its own justify a submission.
