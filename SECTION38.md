--------------------------------------------------------------------------------
Section 38  WE ARE ALREADY RUNNING THE PXR pp50 ARCHITECTURE. WHAT WAS MISSING IS
            THE v4_13d ASYMMETRIC HEAD (2026-09-26)
--------------------------------------------------------------------------------

The PXR README settles which model carried the blend: pp50 = v4_4_aug2_p50, from
train_pec50_pxr_v4_4.py, at 75%. v4_13d is p13d at 15%, and its standalone holdout-30 RAE
(0.5463) is the WEAKEST of the three -- it earns its place by decorrelation, not accuracy.

THE ARCHITECTURES ARE THE SAME. Side by side:

    PXR v4_4._build                      our train_cyp_st_v3._build
    BondMessagePassing(d_h, depth,       BondMessagePassing(d_h, depth,
        dropout, activation='relu')          dropout, activation='relu')
    RegressionFFN(input_dim=hd+n_desc,   RegressionFFN(input_dim=hd+n_desc,
        hidden_dim=hd, n_layers, dropout,    hidden_dim=hd, n_layers, dropout,
        n_tasks=1)                           n_tasks=1, criterion=DeadZoneSpreadMSE)
    MPNN(mp, MeanAggregation(), ffn)     MPNN(mp, MeanAggregation(), ffn,
                                              batch_norm=True, metrics=[...])

Same trunk, same aggregation, same fusion of target descriptors onto the aggregated
embedding via x_d, same readout. THE ADAPTIVE TIERS ARE IDENTICAL TOO -- the CYP2C9 run
printed "hidden=768 depth=6 layers=4 aug=1x" at N=4376, which is PXR's `very_large` tier
exactly. Our two additions are the dead-zone criterion (so a censored row costs nothing
below its bound) and batch_norm.

MOST OF THE RECIPE IS ALSO ALREADY THERE, including the one that looked like the biggest
gap. PXR upweights the potent tail 3x above pEC50 5.5 to stop the model spending capacity
on inactives. We do the same thing in a better-grounded form: interval weighting, which
gives up to 4x to the tightest-measured compounds. Since corr(y, interval half-width) runs
-0.69 to -0.89, the tightest intervals ARE the potent compounds, and the weight comes from
the assay rather than from a threshold someone chose.

WHAT IS GENUINELY ABSENT, after the comparison:

  1. 10-fold against our 5. Not a free change -- artifacts/folds.csv defines 5 folds and
     every existing member is aligned to it, so switching breaks comparability with the
     whole blend. Deferred deliberately.
  2. THE v4_13d ASYMMETRIC CLASSIFICATION HEAD. Now implemented as CYPMultitaskMPNN in
     train_cyp_st_v3.py, behind --multitask_clf.

WHY THAT HEAD IS WORTH MORE TO US THAN IT WAS TO PXR. Dose-response curves were run only
on primary-screen HITS, so training is enriched for actives while the blind set is the
unfiltered library. The measured train -> blind mean shift is +0.64 / -0.23 / +1.86 /
-0.66 on CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4. Inactive over-prediction is not a side effect
here, it is the central distribution problem, and the entire affine placement machinery
exists to correct it after the fact. PXR measured the same thing from the other side: 22%
of test compounds below pEC50 4.0 drove 53% of total RAE.

AND OUR INACTIVE LABEL IS BETTER THAN PXR'S. They had to threshold at pEC50 < 4.0. We have
thousands of MEASURED inactives -- the left-censored screen non-hits, already carried as
lt_mask. A row counts as inactive if it is left-censored OR sits at or below --act_thresh
(default: the assay floor), so the supervision is an experimental result, not a cut.

    total = (1 - clf_weight) * dead-zone regression loss
          +      clf_weight  * BCE(clf_logit, 0) over INACTIVE ROWS ONLY

VERIFIED OFFLINE, with mock graph objects since chemprop needs the GPU box: forward shape;
the asymmetric mask (5 inactive of 12, active rows contributing exactly zero); the FP
penalty rising when logits are pushed up; clf_weight=0 reducing to pure regression; the
dead zone charging 0.053333 against a hand-computed 0.053333; a left-censored row free
below its bound and charged 8.41 above it; predict_step shape; optimizer. NOT verified:
integration with real chemprop batches.

TWO BUGS CAUGHT WHILE BUILDING IT. `json` was never imported in train_cyp_st_v3.py, so the
config writer would have raised NameError on the first run. And a CYPMultitaskMPNN cannot
be rebuilt by MPNN.load_from_checkpoint -- it is not an MPNN -- so its architecture is
written to {iso}_model_config.json at train time and _safe_load reconstructs from it. Same
lesson as the descriptor meta file: a checkpoint that cannot say what it was built from is
a failure waiting for the next PREDICT run.

    python train_cyp_st_v3.py --pseudo --lean_descriptors --multitask_clf --mode BOTH ^
      --model_dir models_cyp_st_v3_mt --oof_output oof_chemprop_st_v3_mt.csv ^
      --output predictions_cyp_st_v3_mt.csv

Expect a DECORRELATED BLEND MEMBER, not a solo winner -- that is what p13d was at 15%.
Score it with scripts/model_audit.py before it goes near a submission.

--------------------------------------------------------------------------------
CORRECTION, same day, after re-reading the README's own evidence
--------------------------------------------------------------------------------

The file identification is confirmed -- README line 297:

    | **v4_4_aug2_p50 (pp50)** | train_pec50_pxr_v4_4.py --patience 50 | 0.4641 |

So v4_4 is the 75% model and the architecture comparison above stands.

BUT THE README ARGUES AGAINST THE HEAD I JUST BUILT, and I should have surfaced this
before writing 150 lines of it. Per-region on the 30-compound holdout:

    region            n   p13d MAE   pp50 MAE   p13d bias   pp50 bias
    Inactive <4.0     7     0.962      0.775      +0.494      +0.312
    Moderate 4-5.5   18     0.295      0.272      -0.056      +0.060
    Potent >5.5       5     0.334      0.266      -0.294      -0.224

p13d LOSES TO pp50 IN EVERY REGION -- including the inactive region the asymmetric head
exists to fix. The README's own words: "p13d's asymmetric BCE reduces inactive bias
relative to its symmetric predecessor, but not below pp50's bias." And the weight history
runs the wrong way for it: 36% in v2 down to 15% in the final, with "increasing pp50 weight
and decreasing p13d weight monotonically improves holdout-30 RAE."

I also called it "decorrelated". It is the LEAST decorrelated of the three:

    pp50 vs p13d     r = 0.953     <- highest
    pp50 vs UniMol   r = 0.919
    p13d vs UniMol   r = 0.918

So the 15% is a compromise between a weak model and a modest diversity gain, not an
endorsement. THE PRIOR FROM PXR'S OWN DATA IS NEGATIVE.

WHY IT MIGHT STILL BE WORTH ONE RUN HERE, stated as a hypothesis rather than an
expectation. PXR's inactive region was 7 compounds of 30. Ours is the dominant feature of
the data: thousands of left-censored screen non-hits, a blind set drawn from the unfiltered
library, and a train -> blind mean shift of +1.86 on CYP2D6. The head's supervision here is
also measured rather than thresholded. An effect that was marginal on 7 compounds could be
real on thousands -- or the same "splits gradient between two objectives" cost could simply
dominate again.

PRIORITY, REVISED. The lean-descriptor run comes FIRST: it is the un-confounded experiment
already set up, and Section 36 gives it a measured positive prior (+0.0400 CYP2C9, +0.0239
CYP2D6). --multitask_clf goes after it, and only if there is GPU time to spare. Do not run
both at once -- that would repeat exactly the confound that made this morning's retrain
uninterpretable.

WHAT THE README SAYS ACTUALLY MADE pp50 THE 75% MODEL, none of which is exotic:
  - patience 50 over 25, worth 0.0005 RAE (0.4646 -> 0.4641). Marginal.
  - high_tail_mult 3.0. Note --high_tail_mult 10.0 scored 0.5359, far WORSE, so the
    optimum is shallow and near 3x. Our interval weighting tops out at 4x, in that zone.
  - the aug2 clean training set, which has no analogue here.
  - 10-fold CV.
The architecture itself, which is the part we were asked to emulate, we already had.
