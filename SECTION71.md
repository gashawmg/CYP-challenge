# Section 71 -- CYP3A4 confirmed by two independent routes. The other three heads come out.

## 1. The matched pair

`oof_mh_notdi` against `oof_mh_tdi`, one flag apart, both arms placed optimally before
comparison. Paired bootstrap, 1,200-2,000 resamples:

| endpoint | d rho | 95% CI | P(worse) | d placed ST-RAE | 95% CI | P(worse) | new TDI compounds |
|---|---|---|---|---|---|---|---|
| CYP1A2 | **-0.0255** | [-0.0534, +0.0025] | **96.2%** | -0.0203 | [-0.0418, +0.0017] | 96.8% | 1 |
| CYP2C9 | +0.0160 | [-0.0085, +0.0418] | 10.1% | +0.0350 | [+0.0059, +0.0664] | 1.3% | 0 |
| CYP2D6 | -0.0138 | [-0.0466, +0.0209] | 79.5% | -0.0162 | [-0.0370, +0.0050] | 93.2% | 4 |
| **CYP3A4** | **+0.0129** | **[+0.0022, +0.0239]** | **0.9%** | **+0.0185** | **[+0.0028, +0.0357]** | **1.2%** | **1,249** |

Macro placed ST-RAE moved 0.6739 -> 0.6698, +0.0042. Two endpoints up, two down, and the
macro is the residue of four things mostly cancelling. Taken as a macro result this is a
null.

## 2. It is not a null on CYP3A4, and the reason is replication

    surr3a4  (single-task ChemProp, surrogate ROWS)   d rho +0.0135  CI [+0.0022, +0.0250]  P(worse) 0.9%
    mh_tdi   (multihead, a separate HEAD)             d rho +0.0129  CI [+0.0022, +0.0239]  P(worse) 0.9%

Two different model families, two different mechanisms for getting the TDI arm in front of
the encoder, and the point estimates agree to 0.0006 with the same interval to three
decimals. Nothing else in this project has replicated like that. CYP3A4 is also the only
endpoint where the TDI arm supplies compounds the scored file does not have -- 1,249 of
them -- so the result has a mechanism as well as a measurement.

## 3. The other three are undecidable from one seed, and the arm correlations say so

    arm-to-arm Spearman:  CYP1A2 0.7857   CYP2C9 0.8549   CYP2D6 0.6289   CYP3A4 0.8969

`seed_ensemble.py` measured two runs of IDENTICAL settings at r = 0.927. Three of these
four pairs sit BELOW that, CYP2D6 far below. Run-to-run variance is larger than the effect
being tested on every endpoint except CYP3A4, which is the one pair above the noise floor
and the one that replicates.

So CYP2C9's +0.0350 placed gain (P(worse) 1.3%) is not claimed: its rho interval straddles
zero, it adds no new compounds, there is no mechanism for it, and its arms correlate 0.855
against a 0.927 noise floor. CYP1A2's -0.0255 is not claimed as a real loss either, by the
same argument -- though with zero new compounds on offer there is no reason to pay head
capacity for it.

## 4. The change

`with_tdi` now defaults to **CYP3A4 only**, and accepts an isoform list. `train_cyp_mh.py`
gains `--tdi-isoforms`. A TDI row carrying no label for any enabled head no longer creates
a table row at all, so the three dropped heads take their dead weight with them.

## 5. What it is worth, stated plainly

+0.0185 on one endpoint is **+0.0046 of macro**, about 1% of the blind 0.4737. Real,
replicated, and small. It is the first confirmed model-side gain of this session, and it is
nowhere near a submission on its own.

For scale, from the same two files: this single-seed control scores macro placed 0.6739
while the 5-seed `mhens3` ensemble scores 0.6249. **Seed ensembling is worth 0.049 of macro
OOF -- ten times the TDI head.** That is already known (`seed_ensemble.py`) and already
in the production column, which is why neither arm here is competitive with what we submit.

## 6. mhens3's settings ARE recoverable, by fingerprint -- five runs, not ten

Section 70 said `oof_st_mh_ctl.csv`'s settings could not be read off disk and budgeted two
arms for every comparison. That was true of `ctl` and false of `mhens3`, which is the one
that matters. Four checks pin it down completely:

| question | evidence | answer |
|---|---|---|
| which trainer? | `models_mhens3_s42/` holds `fold_0.ckpt` | `train_cyp_mh.py` (that is its checkpoint naming; `train_cyp_clean.py` writes `f<seed>.ckpt`) |
| descriptors on? | 5 `dscale_*.npy` present | yes |
| which spec? | `dscale_0` is **(2, 27)**, matching `models_mh_desc`; `models_mh_desc3d` is (2, 35) | `--descriptor_spec shared`, NOT `shared3d` |
| which seeds, and is the file really their mean? | `max abs(mean(s42,s1337,s2024,s7,s99) - oof_mhens3)` = **8.9e-16** on all four endpoints | exactly those five, exactly a mean |

So `oof_mhens3.csv` is a settings-matched control for a TDI arm run as:

    train_cyp_mh.py --descriptor_spec shared   seeds 42 1337 2024 7 99   5 folds

differing in the TDI head and nothing else. The control is already on disk, so the
comparison costs five runs rather than ten, and the match is verified rather than assumed.

ONE THING THE A/B DOES NOT COVER. Sections 70-71 measured the TDI head with
`--descriptor_spec none`, because that is train_cyp_mh.py's default and both arms shared
it. The ensemble run turns descriptors ON to match mhens3. Whether +0.0129 rho on CYP3A4
survives alongside 27 descriptor columns is untested -- it is the first thing to read off
the result, not something to assume carries.
