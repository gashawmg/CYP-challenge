# How the blind compounds are predicted

Written for someone with no prior exposure to this project. It describes what actually
happens, in order, and marks plainly where the pipeline stops being justifiable in the
usual terms of a QSAR study -- representation, splitting, model -- and becomes empirical
tuning against a leaderboard.

---

## 1. The task

Four human cytochrome P450 enzymes: **CYP1A2, CYP2C9, CYP2D6, CYP3A4**. For each, predict
how strongly a compound inhibits it, as **pIC50** (higher = more potent). We are given
4,905 training compounds and 750 compounds whose measured values are withheld. We predict
all four endpoints for all 750 and upload.

Scoring is **MA-ST-RAE**: for each compound the organisers know a credible interval around
the measured value, and a prediction inside that interval costs nothing. Outside it, you
pay the distance to the nearest edge. The result is averaged over compounds, normalised
against the score a constant prediction would get, then averaged over the four endpoints.
Spearman correlation, R2 and MAE are reported alongside but the ranking metric is ST-RAE.

## 2. The data, and the one fact that shapes everything

Two kinds of measurement exist, and the proportions matter:

| endpoint | compounds with a dose-response curve | compounds with "no measurable activity" | censored |
|---|---|---|---|
| CYP1A2 | 1,180 | 2,932 | 71% |
| CYP2C9 | 1,026 | 2,393 | 70% |
| CYP2D6 | 1,364 | 3,012 | 69% |
| CYP3A4 | 2,335 | 707 | 23% |

**Roughly 70% of the training data on three endpoints is not a number -- it is the
statement "below pIC50 4.0".** These are left-censored. They are real information (the
compound was tested and did nothing) but they cannot be fitted as a value.

Curves were only run on compounds that had already shown activity in a primary screen, so
the compounds that *do* have a number are a biased, activity-enriched sample. The blind set
is drawn from the unfiltered library.

We also use **16,233 external compounds** from a public qHTS screen with their own
inhibition readouts for five isoforms.

## 3. Molecular representation

Two things are concatenated and given to the model:

1. **A learned graph embedding.** The molecule is a graph -- atoms as nodes, bonds as
   edges. A directed message-passing network (chemprop's D-MPNN) passes information along
   bonds for several rounds and pools the result into a fixed-length vector. Nothing is
   hand-designed; the representation is fitted to the task.

2. **27 hand-chosen descriptors**, concatenated to that vector before the final readout.
   Every one has to name a physical requirement for binding or turnover in a CYP active
   site in one sentence:
   - *occupancy and shape*: heavy-atom count, fraction sp3, Kappa3, rotatable bonds
   - *recognition*: logP, polar surface area, H-bond donors and acceptors, aromatic rings
   - *electronics*: partial-charge and E-state extremes -- where oxidation happens
   - *ionisation at pH 7.4*: predicted basic and acidic pKa, logD, net charge, zwitterion
     flag. This is the single biggest determinant of CYP2C9 (anion-preferring) versus
     CYP2D6 (cation-preferring) selectivity.
   - *heme coordination*: SMARTS counts for aromatic nitrogen, imidazole, triazole,
     nitrile and relatives -- the classic type-II binders that ligate the iron directly.

   The set was cut down from ~62 columns by removing near-duplicates (eight different
   measures of "how big is it") and anything with no mechanistic reading.

Morgan fingerprints appear in this project **only for similarity analysis**, never as model
input.

## 4. Splitting

- **5-fold cross-validation** over the challenge compounds. Every reported number is
  out-of-fold: the model never saw that compound's label when predicting it.
- **External data is always in training, never in validation.** It is a different assay;
  scoring on it would not tell us anything about the blind set.
- **Leakage guard**: any external compound whose InChIKey connectivity layer matches a
  blind compound is dropped. 162 external rows that were the same structure as a challenge
  compound written differently are merged rather than duplicated.
- **A blind-like holdout** of 394 compounds, chosen so that their nearest-neighbour
  similarity to the remaining training set matches the blind set's similarity to the full
  training set. Ordinary random folds are optimistic because the blind set sits at a
  different distance from what we know.

## 5. The model

**One shared encoder, eighteen output heads:**

| heads | what they predict |
|---|---|
| 4 | the scored challenge pIC50s -- **the only ones ever submitted** |
| 4 | challenge single-concentration screen log2 fold-change |
| 5 | external pIC50, including CYP2C19 which is never scored |
| 5 | external qHTS maximum response |

The fourteen auxiliary heads exist to shape the shared representation and are discarded at
prediction time. Nothing is harmonised across assays -- each keeps its own scale and its
own loss term -- which avoids needing a cross-assay correction we do not have.

**The loss understands censoring.** For a compound recorded as "below 4.0", predicting
anything at or under 4.0 costs nothing; only predicting it as active is penalised. This is
the same one-sided shape the scoring metric uses.

**Five random seeds are trained and averaged.** Two runs of this model with identical
settings correlate only r = 0.79-0.93 depending on endpoint, so a single run carries a lot
of noise that has nothing to do with chemistry.

That is the model. It is describable, the representation is justifiable compound by
compound, and the splitting is honest.

---

## 6. What happens after the model -- and where the justification runs out

The submitted file is **not** this model's output. Between the model and the upload sit
four more layers, added one at a time over six weeks, each because it improved a
leaderboard score:

1. **Eleven "members".** Every model variant ever trained -- different architectures,
   feature sets, and a Uni-Mol 3D model -- is kept as a member.
2. **A greedy blend search, per endpoint.** For each endpoint it picks a subset of members
   and averages their z-scored predictions, choosing the subset by out-of-fold performance
   on a re-weighted evaluation pool.
3. **An affine placement.** The blended z-scores are mapped back to pIC50 using an estimate
   of the blind set's own mean and spread -- an estimate itself *solved backwards from
   three previously scored submissions*.
4. **A per-endpoint splice and spread multiplier.** The final file takes CYP1A2 and CYP2D6
   from one earlier submission and CYP2C9 and CYP3A4 from the blend, then multiplies two
   endpoints' spreads by a constant chosen from measured leaderboard responses.

**Steps 1-5 of this document are a model. Step 6 is curve-fitting to a scoreboard.** It is
not interpretable, it could not be written up, and it cannot be defended to a chemist as
chemistry. The leaderboard has a "Model Report Link" column; ours reads *Not submitted*,
and step 6 is the reason.

## 7. Why it grew, and what the evidence now says about removing it

It grew because each layer was measured and each one paid at the time. But three
measurements this week undercut the whole edifice:

- **74-86% of every member's error is shared with the other members.** They fail on the
  same compounds. Averaging eleven of them removes only the 14-26% that is independent.
- **Improving a member barely moves the blend.** Target descriptors gained +0.0183 macro
  Spearman standalone -- confirmed 20/20 across five seeds and four endpoints -- and
  +0.0016 after blending. Seed-averaging gained +0.13 summed standalone and roughly zero
  after blending.
- **The single model is now nearly as good as the committee.** On CYP3A4 the blend search,
  given the 5-seed descriptor model, selected **that model alone** over any combination,
  and it beat the shipped four-member blend on the evaluation pool.

So the complexity is no longer obviously earning its place. The honest next experiment is
to **score the plain 5-seed multi-head model, with one defensible placement step and
nothing else, against the current spliced submission** -- and find out what the six weeks of
accretion are actually worth. If the answer is "a few thousandths", the pipeline should
collapse back to something that can be written down, and the remaining effort should go
into the representation and the loss, where the error actually lives.

That experiment needs no GPU. It is the next thing to do.
