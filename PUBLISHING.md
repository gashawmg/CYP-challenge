# Pre-publication checklist

This repository is **private**. It is intended to go public after the challenge closes,
as the "Open Code" entry alongside a model report. Work through this before flipping it.

## Before making the repository public

- [ ] **Check the data terms.** `data/` is gitignored. Confirm whether the organisers
      permit redistribution of the challenge CSVs. If not, keep the fetch instructions in
      the README and leave the files out. If yes, decide deliberately rather than by
      default.

- [ ] **Confirm no blind labels are present anywhere.** The blind set must appear only as
      structures. Any file containing a true value for a blind compound cannot be
      published and should not exist.

- [ ] **Re-run the secret scan.** Credentials are currently read from the Hugging Face CLI
      store via `get_token()` and nothing is hardcoded. Verify this still holds:

      ```
      git grep -nE "hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"
      ```

      Check the full history, not just the working tree -- a secret committed once stays
      reachable after deletion.

- [ ] **Review third-party files.** `artifacts/novartis_tdi_predictions.csv` is extracted
      from published supplementary material. Confirm its licence permits redistribution,
      or drop it and leave the extraction script with a pointer to the DOI.

- [ ] **Strip notebook outputs.** `tdi_pipeline.ipynb` carries 26 stored outputs and
      `tdi_shift_target.ipynb` carries 7. Stored outputs can embed rows of data and file
      paths. Clear them, or add an `nbstripout` filter so it cannot be forgotten.

- [ ] **Add a LICENSE.** Not yet chosen. MIT or Apache-2.0 for the code; note separately
      that the challenge data carries its own terms.

- [ ] **Rewrite the README for a stranger.** It currently assumes the reader is the author.
      An outsider needs: what the challenge is, what the approach was, what the result was,
      and how to reproduce one submission end to end.

- [ ] **Decide what to do with the `SECTION*.md` notes.** Roughly fifty working notes,
      written to self. They are the most honest record of the method and also the least
      polished. Either move them to `docs/notes/` with a short preface explaining what they
      are, or distil them into one methodology document.

## What makes this repository worth publishing

Most of the project's value is not the models, which are ordinary. It is the measured
record: ~90 numbered findings, roughly a third of them refutations of things that looked
true. Several results are negative and worth more than the positive ones -- that the blend
keeps about 12% of a member-level gain, that out-of-fold selection does not transfer to the
blind set, that architecture is not the axis of decorrelation.

If the notes are cut to make the repository look tidier, the thing that makes it useful
goes with them.
