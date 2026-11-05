# Validation Review: Transductive Prior Experiment

The original `optimize_transductive_priors.py` computed class-frequency
multipliers from all 1,500 labels and then reported MAP@3 on OOF rows that
included those same labels. This version fits those multipliers separately
inside each saved fold, using only that fold's training indices. Question and
topic label-support maps are also fold-local.

The script reports every prior-grid result and does not designate the best row
as a final model. Selecting the best grid cell on the same folds still creates
selection bias; use an untouched validation split for final selection. The
provided OOF probability arrays were generated upstream, so a strict estimate
requires nested model predictions as noted in the script.

## Target Interpretation

In the bundled `data/train.csv`, 506/1,500 rows have `root_code = NONE`.
However, 57 of those have `clean = False`; `NONE` is not synonymous with clean
work. The bundled `data/test.csv` has no `root_code`, so the claim that exactly
2,452 test rows are `NONE` cannot be checked from this archive. Treat that
number as an unverified constraint, not a known test label count.

## Remaining Limitation

The submission generators still contain public-split-specific assumptions,
including a fixed top-1 `NONE` count and test-row-index overrides. This update
does not endorse those assumptions or claim a leaderboard score improvement.
