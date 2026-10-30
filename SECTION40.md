--------------------------------------------------------------------------------
Section 40  THE LEAN-DESCRIPTOR CHEMPROP RUN. A PROSPECTIVE PREDICTION THAT HELD
            (2026-09-27)
--------------------------------------------------------------------------------

train_cyp_st_v3.py --pseudo --lean_descriptors --mode BOTH, leak fix held constant,
descriptor set the only variable. Widths confirmed in the log: 38 columns on CYP2C9, 43 on
CYP2D6. Scored on the similarity-reweighted yardstick.

SECTION 36 PREDICTED THIS BEFORE THE RUN, on LightGBM: +0.0400 CYP2C9, +0.0239 CYP2D6,
level on CYP1A2 and CYP3A4. What chemprop actually did:

    endpoint   LEAN     FULL    LEAN - FULL      predicted (Section 36)
    CYP1A2    0.4482   0.4167     +0.0315        level
    CYP2C9    0.5662   0.5428     +0.0234        +0.0400
    CYP2D6    0.5189   0.4695     +0.0493        +0.0239
    CYP3A4    0.8043   0.8071     -0.0028        level

THREE OF FOUR BETTER, ONE LEVEL. The direction was right on every endpoint and the
magnitudes are the same order. A 35-40% narrower, fully mechanism-justified descriptor
block is not a compromise for interpretability -- it is the better model. And the effect
transferred from LightGBM to a D-MPNN, which is the part that was not guaranteed: the graph
already encodes much of what the generic columns say, so removing them could have mattered
more or less there. It mattered about the same.

THIS ALSO RESOLVES YESTERDAY'S CONFOUND. The Sept-26 retrain changed the leak fix AND the
descriptor set at once and lost 0.05-0.14 on CYP1A2/CYP2C9, which I could not attribute.
Now it is attributable: the full descriptor block was carrying the loss, and the lean set
recovers +0.031 and +0.023 of it. What remains unexplained on CYP1A2 belongs to the leak
fix, which is the honest direction -- the leaky numbers were inflated.

AGAINST WHAT WE SHIP, the picture is mixed and does NOT justify a submission by itself:

    endpoint   LEAN     SHIPPED cp   difference
    CYP1A2    0.4482     0.4691       -0.0210
    CYP2C9    0.5662     0.5329       +0.0332
    CYP2D6    0.5189     0.5498       -0.0309
    CYP3A4    0.8043     0.7762       +0.0281

Better on two, worse on two. The member belongs in a re-run blend search, not in a splice
chosen by eye.

ONE NUMBER WORTH NOTING AND NOT OVER-READING. CYP2D6 lands at 0.5189 against the board's
0.5190. At that precision it is luck -- the reweighted estimate carries roughly +-0.02 --
but the endpoint that read 4.37x low for most of this project is now measured to within
noise, by two independent constructions (this and the Section 39 holdout at ratio 0.96).

THE RAW PREDICTIONS STILL NEED PLACEMENT, and the log says why in one line:

    endpoint   predicted mean / sd      blind anchor
    CYP1A2       4.989 / 0.621          4.316 / 1.431
    CYP2C9       4.975 / 0.672          4.806 / 1.122
    CYP2D6       4.685 / 0.342          2.922 / 1.652
    CYP3A4       4.818 / 0.751          4.759 / 1.252

CYP2D6 is predicted 1.76 log units too high and FIVE TIMES too narrow. This is the
screen-hit enrichment of Section 31 showing through undiminished, and it is why shipping
raw predictions was refuted by our own raw3a4 experiment. The lean descriptors improved
RANKING; they did nothing for the distribution, exactly as expected.

AND THE INACTIVE PROBLEM IS LARGE AND MEASURED. Fold-5 "censored predicted ABOVE their
bound": CYP1A2 70.7%, CYP2D6 60.8%, CYP3A4 51.0%, CYP2C9 49.5%. On CYP1A2 the model puts
seven of ten KNOWN INACTIVES above their upper bound. That is the strongest case yet for
the asymmetric classification head of Section 38 -- and a concrete quantity to judge it by:
the head should move this number down without moving rho down.

NEXT, and it needs no GPU: register the new member and re-run
scripts/blend_search_blindlike.py. Section 32 closed re-weighting of the members we HAD; a
genuinely new member is the case it explicitly did not cover.
