# SOMOS prospective-validation feasibility note

Status on 2026-08-29: the SOMOS v2 protocol is locally frozen at SHA-256
`81daeb5dbfcac387ea9bad14dffe0603715999524028a2902757fc6aa1c241d9`.
No SOMOS audio, labels, item values, or cached target scores had been retrieved
at freeze time. GitHub commit
`d3b1dc01b70486d67183d64dee3a0680cb9961b7` now provides the independent
timestamp, so the retrieval gate is satisfied.

## Feasible fixed bank

The existing Scenario A bank is suitable for a prospective external test
because it contains no locally trained D3 or U3-HuBERT student. It consists of
ten public inference runners and 27 score outputs:

- DNSMOS P.835: overall and signal
- DNSMOS P.808: MOS
- SQUIM objective: PESQ, STOI, and SI-SDR
- NISQA and Distill-MOS: one MOS output each
- SCOREQ: natural and synthetic outputs
- UTMOS: MOS
- SIGMOS: seven quality dimensions
- Audiobox Aesthetics: four dimensions
- Uni-VERSA: five predicted quality metrics

The exact member order is already machine-defined by Scenario A in
`scripts/stack_feasible.py`. Before activation, pin each runner revision,
weight hash, preprocessing configuration, and score-column name in the frozen
protocol. Do not substitute a newer checkpoint or add a predictor after any
target label is retrieved.

## Activation sequence

1. Completed: project history showed no earlier SOMOS artifact, label cache, or
   SOMOS-specific analysis.
2. Completed: public metadata fixed SOMOS v2, `split1/clean`, the clean MOS
   target, and the sentence-derived source grouping rule.
3. Completed: `third_corpus_protocol_frozen.md` fixes the 27-output bank, six
   controls, official train/validation/test partitions, train-fitted ECDF
   ranks, budgets 200/500/1,000/2,500/full, seeds 0 through 9, and the primary
   source-cluster bootstrap comparison.
4. Completed: `scripts/freeze_validation_protocol.py` created the
   non-overwriting SHA-256 sidecar, and the two frozen files were archived in
   GitHub commit `d3b1dc01b70486d67183d64dee3a0680cb9961b7`.
5. Retrieve the one frozen dataset revision, record its endpoint, revision,
   time, byte size, row count, and SHA-256, and preserve the raw provenance.
6. Generate the 27 scores with the frozen runners, validate exact ID alignment,
   no duplicate IDs, finite columns, and the declared complete-case rule. Only
   then run the frozen analysis once, retaining all outputs and failures.

## Feasibility and limits

This is computationally feasible but is not a no-compute experiment: 10
inference runners must score the selected audio split. No GPU job should start
until the protocol is frozen and an execution budget is agreed. A cache of
public predictor columns would remove this inference burden only if its
provenance and exact revisions can be verified before label retrieval.

The result would estimate transfer of a fixed, public zero-shot predictor bank
and a label-fitted linear combiner. It would not prove that public predictors
were never pretrained on SOMOS or a related corpus. Such overlap is generally
not auditable and must be disclosed. It also would not validate the present
29-output bank, locally trained students, or any post-retrieval method change.
