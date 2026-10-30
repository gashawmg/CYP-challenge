# Section 61. The alerts are literature-grounded. What they lack is not chemistry, it is context.

Run date 2026-10-12. The question was whether the TDI feature set was built from the
literature or assembled by throwing everything at the model. Answer: the alert block is
properly grounded, the individual alerts carry real and chemically sensible signal, and
the thing it is missing is measurable and specific.

---

## 1. What is already there

`cypkit/tdi_alerts.py` is 51 SMARTS-derived columns and its header states the mechanism
explicitly: TDI is bioactivation, not potency. Its coverage is the standard
mechanism-based-inactivation vocabulary:

- **MI-complex precursors** -- N,N-dimethyl tertiary amine, N-methyl secondary amine,
  benzylic N, piperidine, piperazine, morpholine. These are the alkylamines that
  N-dealkylate to a nitroso species and coordinate the heme iron (erythromycin,
  diltiazem, verapamil, fluoxetine).
- **Carbene precursors** -- methylenedioxyphenyl.
- **Terminal and internal alkynes**, the classic suicide substrates.
- **Oxidisable heterocycles** -- furan, thiophene, pyrrole, thiazol-2-amine, 3-methylindole.
- **Quinone / quinone-imine precursors** -- para-open phenol, unblocked aniline, anilide,
  catechol, aryl methoxy.
- **Direct electrophiles** -- epoxide, Michael acceptor, thiourea, hydrazine,
  cyclopropylamine, nitroaromatic, alkyl halide, activated aryl halide.

That is the Kalgutkar-style alert list, and it is not an accident -- it is the right list.

## 2. The individual alerts work. Measured on the training labels.

P(TDI | alert fires) minus P(TDI | it does not), on the labelled rows:

**CYP3A4** (base rate 0.213)

| alert | n | P(TDI\|hit) | lift |
|---|---|---|---|
| cyclopropylamine | 54 | 0.537 | **+0.329** |
| morpholine | 157 | 0.382 | +0.177 |
| terminal alkene | 26 | 0.385 | +0.173 |
| phenol, para open | 40 | 0.375 | +0.164 |
| carbamate | 33 | 0.364 | +0.152 |
| pyrrole | 178 | 0.292 | +0.083 |
| sulfide / thioether | 230 | 0.052 | **-0.172** |
| acyl-glucuronide acid | 117 | 0.026 | **-0.194** |

**CYP2D6** (base rate 0.216)

| alert | n | P(TDI\|hit) | lift |
|---|---|---|---|
| furan ring | 47 | 0.404 | **+0.194** |
| furan, substituted | 45 | 0.378 | +0.166 |
| pyrrole | 102 | 0.324 | +0.115 |
| **reactive ring AND basic N** | 95 | 0.316 | +0.106 |
| thiophene | 98 | 0.306 | +0.096 |
| aryl methoxy | 144 | 0.139 | -0.086 |

Cyclopropylamine at +0.33 is textbook -- the ring opens under oxidation to a reactive
radical and the enzyme is dead. The negative lifts are equally mechanistic: a carboxylic
acid headed for glucuronidation is not going to be bioactivated by a P450, and the model
has learned that from 117 compounds.

13 of 38 usable CYP3A4 alerts and 15 of 31 CYP2D6 alerts move the rate by more than 0.05.
This is not a dead feature block.

## 3. The actual failure: presence is saturated

| | positives carrying any reactive alert | negatives | lift |
|---|---|---|---|
| CYP3A4 | 0.572 | 0.527 | **+0.045** |
| CYP2D6 | 0.577 | 0.558 | **+0.019** |

**Fifty-seven per cent of the positives carry a reactive alert -- and so do fifty-three
per cent of the negatives.** On a drug-like library almost everything has a furan, an
aniline, an amine or a thioether somewhere. Asking "is there a liability in this
molecule" has almost no answer on this data.

That is the gap, and it is not a gap in chemical knowledge. The alert header already
says what is needed -- *"a furan is only a liability if it can be oxidised, and an aniline
only if the ring is not already blocked"* -- and a handful of features pair a motif with
its substitution. But the block stops at presence-plus-substitution. It never asks
**where the alert sits in the molecule, or whether the enzyme can reach it.**

## 4. What the literature says to add, in order of expected value

**(a) Reactivity x accessibility, not presence.** This is the SMARTCyp idea
([Rydberg et al., 2D SMARTCyp](https://pmc.ncbi.nlm.nih.gov/articles/PMC4025842);
[J Cheminform 6:29](https://jcheminf.biomedcentral.com/track/pdf/10.1186/1758-2946-6-29)):
a site's score is a precomputed activation energy for its atom environment MINUS an
accessibility term that is purely topological -- how far the atom sits from the molecule's
2D centre. The accessibility half needs no new software at all, and multiplying each
existing alert by the accessibility of the atom that matched it converts 51 binary flags
into 51 graded ones. Isoform-specific SMARTCyp variants for 2D6 and 3A4 exist.

**(b) The CYP2D6 pharmacophore as a distance, not a co-occurrence.** 2D6's substrate
rule is a protonatable basic nitrogen roughly 5-7 A from the site of oxidation, anchored
by the Glu216 / Asp301 carboxylates
([Paine et al. on E216/D301](https://research.lstmed.ac.uk/en/publications/residues-glutamate-216-and-aspartate-301-are-key-determinants-of--2/)).
The block already has `reactive_ring_and_basic_n`, and it is the fourth-strongest CYP2D6
alert at +0.106 -- but it is a co-occurrence flag. The literature's claim is about a
DISTANCE. Topological shortest path from each basic nitrogen to each alert atom is a
few lines of RDKit, needs no conformers, and is the single most specific untried feature
for the endpoint that is failing.

**(c) Basicity, for the MI-complex amines.** MI-complex formation needs the amine
protonated at physiological pH. The block counts amine substitution patterns but carries
no pKa proxy, so a morpholine (pKa ~6.5, largely neutral) and a piperidine (pKa ~11) are
distinguished only by their SMARTS. A cheap computed basic-pKa estimate separates them.

**(d) Blocked-position features, generalised.** Done ad hoc for aniline and phenol;
worth doing for every oxidisable ring.

## 5. The honest counter-evidence

On CYP2D6, the alert-only model (`lgb_alert`, cyp + tdialert) scores **AUC 0.5635**, while
plain Tanimoto similarity on FCFP scores **0.6283**. Interpretable features are currently
LOSING to raw similarity on that endpoint by 0.065 AUC. On CYP3A4 the picture reverses:
`lgb_alert` reaches 0.7586 of the full model's 0.8200 from 101 interpretable columns, and
the 2,215 fingerprint and descriptor columns add only +0.061.

So the two endpoints answer the question differently. **CYP3A4 is largely explained by
interpretable chemistry already.** CYP2D6 is not, and the reason may be exactly (b): if
2D6 TDI is governed by binding geometry rather than by which reactive group is present,
then neither an alert vocabulary nor a fingerprint will find it, and a distance feature
might.

## 6. What I would do, and what I would not

Build (a) and (b) as one feature block of maybe 30 columns, add it to the pool, and read
two numbers: its own AUC per endpoint, and its rank correlation with `knn_fcfp` on CYP2D6.
Section 54 established that a member pays by disagreeing -- `svc_avalon` gained +0.050 MCC
at correlation 0.369 while scoring BELOW the best single model. A mechanistic block has
every reason to disagree with a fingerprint, which is the strongest argument for trying it.

I would not expect it to replace the fingerprints, and on this project's record I would
not promise it will clear the gate. The CYP3A4 column stays frozen at the 0.4470 blend
whatever this produces, per Section 60.

Two caveats on the literature itself: I could not reach the full text of the FDA TDI QSAR
paper ([PMC11860084](https://pmc.ncbi.nlm.nih.gov/articles/PMC11860084)) or the CYP3A4 TDI
deep-learning paper ([Chem Res Toxicol 2023](https://pubs.acs.org/doi/10.1021/acs.chemrestox.3c00305)) --
both were blocked. The second is worth getting by hand, because it reports how the model
compares against the TDI assay's own experimental variability, which is the ceiling this
whole track is working against and a number we do not have.

Recorded because the premise of the question was that we had thrown everything at the
model without reading, and the truth is narrower and more useful: the reading was done,
the vocabulary is right, and what is missing is the geometry.
