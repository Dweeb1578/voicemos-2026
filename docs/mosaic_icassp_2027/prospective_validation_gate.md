# Prospective third-corpus validation gate

This file defines the gate for a genuinely prospective validation experiment.
It is not itself the frozen corpus-specific protocol. No target labels or
cached target values may be retrieved until every activation item below is
complete.

## Candidate eligibility

A candidate corpus is eligible only if all of the following are true:

1. Its target labels and item-level target values have not previously been
   opened, downloaded, analyzed, or used for model or ensemble selection in
   this project.
2. It was not used to train D3, U3-HuBERT, or another locally trained member in
   the fixed evaluation bank.
3. The project history and current workspace contain no earlier item-level
   label cache from the selected evaluation split.
4. The corpus exposes an auditable grouping unit, such as source utterance,
   speaker, or system, that matches a declared generalization estimand.
5. Required predictor scores already exist as public cached columns, or can be
   generated without changing the declared predictor set after labels are
   opened.

Pretraining overlap of public zero-shot predictors is not generally auditable.
Known or suspected overlap must be disclosed, not silently treated as absent.

## Activation record required before retrieval

Create `third_corpus_protocol_frozen.md` containing all of the following:

- Dataset repository, immutable revision, configuration, and split.
- Exact target column and allowed range.
- Exact predictor columns or exact locally generated predictor artifacts.
- Complete-case and exclusion rules.
- Group identifier and the generalization estimand it represents.
- Baselines: best single, equal ranks, raw ridge, rank ridge, rank NNLS, and
  sparse rank lasso.
- Outer and inner fold counts, seed list, rank-transform scope, and SRCC
  aggregation rule.
- Label budgets and acquisition-order seeds.
- Primary comparison and uncertainty summary.
- Rules for failed downloads, missing columns, constant predictors, duplicated
  IDs, and groups that are too small for the declared folds.
- A statement that no method, feature, hyperparameter grid, budget, or exclusion
  will be changed after target retrieval without labeling the change post hoc.

Then perform these steps in order:

1. Compute SHA-256 of the frozen protocol.
2. Record the hash, local timestamp, Git commit if available, and current
   workspace revision in `third_corpus_protocol_frozen.sha256.json`.
3. Commit or otherwise timestamp the protocol in an append-only remote or
   institutional archive.
4. Retrieve the exact declared split and record endpoint, immutable dataset
   revision, time, byte size, row count, and SHA-256 in a provenance sidecar.
5. Run the declared analysis once and retain all outputs, including failures.

## Permitted post-retrieval changes

Implementation bug fixes are permitted only when the failure and correction are
logged. The original result must remain archived. A corrected run must receive a
new result identifier and explain whether the correction could change the
scientific conclusion.

Exploratory analyses are permitted after the primary run, but they must be
stored separately and labeled post hoc in the manuscript. They cannot replace
the frozen primary result.

## Current status

Activated for SOMOS v2 on 2026-08-29. The corpus-specific protocol is frozen
locally at SHA-256
`81daeb5dbfcac387ea9bad14dffe0603715999524028a2902757fc6aa1c241d9`.
No third-corpus audio or target labels had been retrieved when the hash was
recorded. The exact protocol and sidecar were independently timestamped in
GitHub commit `d3b1dc01b70486d67183d64dee3a0680cb9961b7` on branch
`codex/somos-v2-protocol-freeze`; that commit contains only those two files.
The retrieval gate is satisfied.
