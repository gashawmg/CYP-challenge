# Section 77 -- The cost-weighted yardstick: correct, shipped, and SMALLER than Section 76's framing claimed

## 1. What was claimed, and the correction

After blend4tdi lost I wrote that we had "been selecting models on a yardstick that
disagrees with the board by 0.047 of rho -- larger than every improvement found this
session put together", and framed that as the explanation for being stuck. The first half
is true. **The conclusion was overstated, and the measurement that shows it was available
in the same table.**

The 0.047 drop is nearly UNIFORM across models:

| model | plain rho | cost-weighted rho | drop |
|---|---|---|---|
| mhtdi5 | 0.6516 | 0.5995 | -0.0521 |
| mhens3 | 0.6453 | 0.5912 | -0.0541 |
| v6 | 0.6128 | 0.5514 | -0.0614 |
| cp | 0.5895 | 0.5240 | -0.0655 |
| unimol | 0.5661 | 0.4946 | -0.0715 |

A ruler that is wrong by nearly the same amount for everything still ORDERS correctly,
and ordering is all model selection uses. That is why re-scoring everything on disk
rescued nothing, and it means the old yardstick is not why we are stuck.

## 2. What the new yardstick is actually worth

Across ten OOF columns, how well each ruler predicts the placed ST-RAE those columns
achieve:

    corr(plain rho,       placed ST-RAE) = -0.9758
    corr(P(outside) rho,  placed ST-RAE) = -0.9879
    corr(1/width rho,     placed ST-RAE) = -0.9879

Real, and small. Concretely it repairs two inversions out of ten:

| pair | plain rho says | placed ST-RAE says | cost-weighted rho says |
|---|---|---|---|
| v6 vs mh_1seed | v6 better (0.6128 > 0.6100) | mh_1seed better (0.6600 < 0.6707) | mh_1seed better -- **correct** |
| mt1 vs clean2 | mt1 better (0.5105 > 0.5059) | clean2 better (0.7517 < 0.7618) | clean2 better -- **correct** |

Worth having. Not worth twelve ranks, and I should not have implied it was.

## 3. The weight, and the Section 69 correction that stands

Section 69 rejected 1/width because the cost is U-shaped. That observation was right; the
conclusion (do not weight) was wrong. Two different quantities:

    total cost       where the bill falls -- U-shaped, because we miss the extremes so
                     badly that even a wide interval does not save us
    marginal charge  what ONE unit of error costs -- 1 outside the interval, 0 inside

Selection should follow the marginal charge: `w = P(|e| > h) = 2(1 - Phi(h/s))` for
half-width h and error scale s. That is high for narrow intervals AND for wide-interval
compounds we miss badly -- both arms of the U, which 1/width only half captures.

In practice the two weightings score identically to four decimals here (-0.9879 both), so
the theoretical improvement is not yet measurable. `--weight` exposes both.

**s is fitted ONCE from a reference column and applied to every model in the run.** Letting
each model supply its own s grades each on its own curve -- a model with huge errors would
declare every compound expensive and score itself generously. The reference is printed on
every run so a comparison cannot be quoted without it.

## 4. What is still unexplained, which is the honest state

The conversion gap is real and large and I do not have a mechanism for it.

| | rho | ST-RAE | absorption |
|---|---|---|---|
| furtivepeony (rank 11) | 0.7554 | 0.4348 | 0.6635 |
| **us (rank 47)** | **0.7558** | **0.4737** | **0.7234** |
| preheat-to-450 (rank 1) | 0.7844 | 0.3706 | 0.5975 |

Their ranking is WORSE than ours and they score 0.0389 better; 101% of that gap is
conversion. Against rank 1, conversion is 80% of the gap. Our absorption is the worst of
all twelve teams visible.

And yet: global placement is provably optimal (Section 69), the cost-weighted ranking is
nearly identical to the plain one (this section), and post-hoc uncertainty is dead
(Sections 72-73). Every mechanism this project has proposed for the conversion gap has been
measured and closed. **Something is producing 0.06 of absorption that we have not
identified.**

Do not let the yardstick change paper over that. It is the open question.

## 5. One lead not yet pulled

The leaderboard's column selector offers **"Proprietary Data"** and **"Open Code"** flags,
neither of which we have ever read. If the teams at 0.37-0.43 are flagged as using
proprietary data, a large part of a 0.06 absorption gap stops being a modelling mystery and
becomes a data-access fact -- which changes what is worth attempting with the slots left.
That is one screenshot away and should be checked before any further modelling.
