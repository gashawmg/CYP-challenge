# Section 62. Two papers, and both of them say what yesterday's measurement said

Run date 2026-10-12. Gashaw supplied the FDA (Q)SAR paper (Faramarzi et al., Front
Pharmacol 2025, 10.3389/fphar.2024.1451164) and the supporting information for the
Novartis CYP3A4 TDI deep-learning paper (Fluetsch et al., Chem Res Toxicol 2023,
tx3c00305). Between them they confirm Section 61's diagnosis from an independent source,
name eight alerts we do not have, and hand over a model recipe we have never run.

---

## 1. The FDA profiler alone: sensitivity 92%, specificity 7%

The FDA team curated twelve key publications (Fontana 2005; Kalgutkar 2005, 2007, 2008,
2012, 2015, 2020; Correia 2005; Stepan 2011; Orr 2012; Yu 2015) into **58 MBI structural
alerts**, encoded them as an expert rule set, and qualified them against a 620-compound
CYP3A4 TDI dataset. Then they scored the profiler on its own, external validation:

| | Sens | Spec | PPV | NPV | Concordance | Coverage | chi-sq |
|---|---|---|---|---|---|---|---|
| **SAR profiler alone, CYP3A4 MBI** | **92%** | **7%** | 56% | 40% | 55% | 16% | **0.02** |
| QSAR (TDI) CYP3A4, cross-validation | 80% | 81% | 80% | 81% | 81% | 89% | 206 |
| QSAR (TDI) CYP3A4, external | 61% | 82% | 85% | 56% | 69% | 86% | 13 |

A chi-squared of 0.02 on the alert profiler means no association at all. It flags almost
everything and discriminates nothing -- **by itself**.

That is the same finding as Section 61, reached independently with a different alert set
on a different library:

| | | positives flagged | negatives flagged |
|---|---|---|---|
| ours, 24 reactive alerts, CYP3A4 | | 0.572 | 0.527 |
| ours, CYP2D6 | | 0.577 | 0.558 |
| FDA, 58 alerts, CYP3A4 MBI | | 0.92 (sensitivity) | 0.93 (1 - specificity) |

So the saturation is not an artefact of our SMARTS or of the OpenADMET library. It is what
a presence/absence alert vocabulary does on drug-like chemistry, and the FDA's own use of
it reflects that: they apply the profiler **only to break ties when the QSAR is equivocal**,
never as a classifier.

Our pipeline already uses the alerts the right way -- as 51 columns inside a learner rather
than as a rule -- and `lgb_alert` (cyp + tdialert, 101 columns) reaching AUC 0.7586 on
CYP3A4 is a respectable result in that light.

## 2. Eight alerts the FDA qualified that we do not have

Table 4 of the FDA paper lists the alerts that survived statistical qualification on their
TDI set. Checked against `cypkit/tdi_alerts.py`:

| FDA alert | have it? |
|---|---|
| Benzodioxoles (methylenedioxyphenyl), no C8 substitution | yes (`methylenedioxy_aryl`) |
| Furans | yes |
| Terminal (omega, omega-1) alkynes | yes |
| Epoxides | yes |
| Cyclopropylamines | yes -- **our strongest CYP3A4 alert, +0.329** |
| **Primary amines (non-aromatic)** | **no** |
| **Secondary CYCLIC amines (non-aromatic), as a class** | partial -- piperidine and morpholine only |
| **Piperazines as tertiary amines** (not just aryl) | partial -- `piperazine_aryl` only |
| **Hydroquinone, ortho AND para** | partial -- `catechol` is ortho only |
| **Alkylphenols, quinone-METHIDE precursors** | **no** -- distinct from quinone-imine |
| **Alkyl aromatic ethers ortho/para, quinone-methide precursors** | **no** |
| **Reactive arenes** (arene + alkylamine, arene + alkene) | **no** |
| **Arenes, benzopyran analogues** | **no** |

And Table 3's top CYP3A4 TDI features by Z-score, none of which we encode:

| feature | Z | n |
|---|---|---|
| alkylhalobenzene | 8 | 114 |
| alkoxy-4-alkylbenzene | 6 | 83 |
| alkoxy-4-(oxopropyl)-benzene | 6 | 41 |
| 1,2-diphenoxyethane | 6 | 33 |
| N-cyclopropyl-N-methylformamide | 5 | 57 |

The quinone-methide family is the clearest hole. We have the quinone-IMINE route
(aniline, anilide) and the catechol route, but not the **para-alkyl phenol or para-alkoxy
alkylbenzene to quinone-methide** route, which the FDA ranks among its top five CYP3A4 TDI
features twice over. Our own `phenol_para_open` already scores +0.164 lift and
`methoxy_aryl` scores **-0.086** on CYP2D6 -- the latter is plausibly a methoxy alert that
is too broad, firing on ethers that are not ortho/para to an alkyl group.

## 3. CYP2D6: the paper independently points at the amine pharmacophore

The FDA's top five CYP2D6 features, by Z-score, are **all basic-amine motifs**:
aminomethyl-benzene (Z 18, n 864), secondary alkylamine (16, n 1143), amino-arylethane
(13), secondary amine (13, n 1324), alkyl-aminomethyl-benzene (13). Those Z-scores are the
highest in the entire paper -- CYP3A4's top feature reaches 13, CYP2D6's reaches 18.

They also note that CYP2D6 inhibitors are **less polar on average** than the other
isoforms' and tie it to the hydrophobic 2D6 binding pocket (VandenBrink et al., 2012).
That is two independent arguments for the same Section 61 proposal: for CYP2D6, encode
the basic amine and its geometry, plus PSA-normalised polarity -- not more reactive-group
flags.

Note the caution: these are CYP2D6 **reversible** inhibition features, not TDI. The FDA
had no viable CYP2D6 TDI model -- their CYP2D6 TDI database "did not yield viable models".
Which is itself informative about the endpoint we are stuck on.

## 4. The Novartis recipe, which we have never run

The supporting information gives their exact chemprop call:

```
chemprop_train --dataset_type regression
  --target_columns "log_kobs" "pIC50_CYP3A4" "pIC50_CYP2C9" "pIC50_CYP2D6"
  --split_type scaffold_balanced --split_sizes 0.9 0.1 0
  --epochs 30 --num_folds 1 --ensemble_size 10
```

Three things in one line, and we have done none of them together:

1. **TDI as REGRESSION on a continuous rate, not classification.** They predict
   `log_kobs`. Our continuous analogue, from Section 52's exact label rule, is the
   IC50 shift `pIC50_TDI_condition - pIC50_direct_inhibition`.
2. **Multi-task with the direct-inhibition pIC50s as auxiliary tasks.** We have all four,
   and 3,583 CYP3A4 / 1,497 CYP2D6 rows of the TDI-condition value as well.
3. **A 10-model ensemble.** Our shift experiments were single models.

Section 52 tested shift regression and got Pearson 0.244 on CYP3A4 and 0.057 on CYP2D6 --
but as a single HGB on 729 columns, not as a 10-member multi-task D-MPNN with the pIC50s
carrying the representation. That is a genuinely different experiment and the one the
published work actually uses.

One threshold difference worth noting: the FDA's TDI positive criterion is an **IC50 fold
shift >= 1.5**; the OpenADMET label rule recovered in Section 52 uses **2.0** (log10(2)).
Different studies, different cutoffs, same quantity underneath -- which is another argument
for modelling the shift and thresholding afterwards rather than learning the binary.

## 5. What I would do with this, ranked

1. **Add the quinone-methide family and the four missing amine classes** to
   `tdi_alerts.py`: primary non-aromatic amine, secondary cyclic amine as a class,
   tertiary piperazine, para-hydroquinone, para-alkylphenol, ortho/para alkyl aromatic
   ether, alkylhalobenzene, 1,2-diphenoxyethane. Roughly eight to ten SMARTS, an hour,
   no new dependency. Measure each one's lift the way Section 61 did before adding it to
   any model.
2. **Then Section 61's geometry**: accessibility weighting and the 2D6 basic-N-to-alert
   topological distance. The FDA's CYP2D6 Z-scores make this more attractive, not less.
3. **The Novartis multi-task shift regression**, as a chemprop run on our own data with
   `ensemble_size 10`. This is the biggest single untried idea and it is a GPU evening,
   not an afternoon.

The constraint from Section 60 is unchanged: the CYP3A4 submission column stays frozen at
the 0.4470 blend regardless of what any of this produces locally, and nothing ships on an
out-of-fold number.

Recorded because the FDA paper measured the same saturation we did, on a different alert
set and a different library, and because the Novartis SI contains the one experiment this
project has reasoned about repeatedly and never actually run.
