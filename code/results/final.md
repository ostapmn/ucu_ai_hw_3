# Final selection

Both candidates are stacks (gradient boosting on validation rows) over the same family of inputs. They were selected by group-disjoint nested cross-validation on the dev part of validation and by the lock set, not by the public leaderboard.

## candidate_1

- Stack: `headtail_abcdg_win3_blog`
- Submission file: `code/artifacts/submission_abcdg_win3_blog.csv`
- Dev CV weighted macro-F1: 0.9660 (macro 0.9681)
- Lock weighted macro-F1: 0.9612 (macro 0.9642)
- Inputs: head+tail logits of five encoders (A, B, C, D, G), the head-512 reference encoder, TF-IDF baseline, TF-IDF pool views (full, head, tail), stylometry, drift features, window-scan features of encoders A and G, and the blog-domain specialist logits.
- Meta model: histogram gradient boosting, chosen per fold between three candidates by inner CV.

## candidate_2

- Stack: `headtail_abcd_win`
- Submission file: `code/artifacts/submission_abcd_win.csv`
- Dev CV weighted macro-F1: 0.9607
- Lock weighted macro-F1: 0.9607
- Public leaderboard score of this file: 0.96148 (recorded after the fact, not used for selection).
- Inputs: head+tail logits of encoders A, B, C, D, the reference encoder, TF-IDF baseline, pool views, stylometry, drift features and the window scan of encoder A.

## reproduced

The meta stage of both candidates was rerun from the saved encoder outputs with the same code and seeds. The out-of-fold probabilities, the lock probabilities and the test probabilities are identical to the saved ones (maximum absolute difference 0.0), the recorded dev and lock scores are identical, and the regenerated submission files are byte-identical to the saved ones. The encoders themselves were not retrained in this check; their logits are the saved GPU outputs listed below.

During the check one irreproducibility was found and fixed: the slower boosting variant had been added to the shared candidate list after `headtail_abcd_win` was produced, which changed its result on rerun. The variant is now offered only to the stacks that used it (`SLOW_TAGS` in `code/stack_v2.py`).

## Leakage and overfit checks that were run

- Training sets of every model (`part2`, `part3`, blog specialist) share no id, no prefix key and no exact text with validation; dev, lock and the five folds share no key.
- Shuffled-label control: stack macro-F1 0.27 (chance level), so no feature carries the answer.
- The meta model fits its own training rows perfectly (in-sample 1.0) but out of sample it scores 0.961 to 0.966; the out-of-sample numbers are the ones reported above.
- Remaining optimism: the best of about 25 stacks was picked on the same validation rows, estimated at 0.001 to 0.002.
- 994 validation rows (2.4%) share their last 200 characters with training rows (almost all fusion); test has almost none. These rows are not easier for the model, and without them the blog specialist gain stays at +0.0018.

## Files

- Splits: `code/artifacts/split.csv`, built by `code/make_split.py`
- Encoder training: `code/train_encoder.py`, `code/windows.py`; logits in `code/artifacts/headtail_{a..h}_{val,test}.npy`, `code/artifacts/blog_spec_{val,test}.npy`
- Window scan: `code/window_scan.py`; features in `code/artifacts/winscan_{a,g}_{val,test}.npy`
- CPU models: `code/baseline.py`, `code/baseline_pool.py`, `code/features.py`
- Stacking: `code/stack_v2.py`; scores for all stacks in `code/results/stack_v2.json`; selection in `code/results/stack_v2_final.json` (from `code/make_final.py`)
- Explainability: `code/explain.py`, outputs `code/results/explain_*` and `images/xai_*.png`
- Checks: `code/verify.py` and `code/checks/`
