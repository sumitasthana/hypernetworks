# Reported experiment observations

Read [the experiment log](../EXPERIMENT_LOG.md) for the complete narrative,
earlier learning runs, decisions, limitations, and code findings.

| File | Contents |
| --- | --- |
| [forgetting_traces.csv](forgetting_traces.csv) | All 197 supplied accuracy observations for E08-E14, including each starting row. E08/E09 include supplied losses; E14 includes all 50 loss pairs from saved notebook output. |
| [e13_sampled_losses.csv](e13_sampled_losses.csv) | Six separately supplied E13 loss observations. |
| [notebook_saved_outputs.txt](notebook_saved_outputs.txt) | Text outputs preserved from notebook commit `4407978`, including E14 loss trace, environment setup and E13 sampled losses. |
| [manifest.json](manifest.json) | Intended settings, evidence limits, reported artifact locations, and derived screening summaries. |

These files transcribe the conversation tables and the saved notebook outputs.
The latter corroborate E13/E14 settings and add E14 loss components. The
notebook contains outputs from multiple runs; do not assign every cell to E14. Original Colab
JSON files and model checkpoints are not included or verified here. Earlier
runtime-local files may have been lost when sessions ended. Drive paths record
reported locations, not a guarantee that those files currently exist.

Accuracy is in percent. Drift is an absolute change in percentage points from
the initial task 0 accuracy of 44.6%. Loss values are measured before an update;
accuracies are measured after it. Step 0 is the pre-forgetting model. Blank CSV
cells mean the measurement was not supplied, not zero. Values retain the
precision of the supplied tables. Totals are not reconstructed from rounded
components; separately rounded terms can differ from reported totals by 0.01.

`passes_screen` is derived from both initial accuracies being at least 25%,
target accuracy at most 12%, retained absolute drift strictly below five points,
and step greater than zero. All seven traces start at 26.0% and 44.6%; none
passes the joint screen. This is a diagnostic screen, not proof of erasure.

The manifest distinguishes intended settings from independently verified
metadata. It does not invent missing commit hashes, device details, timings,
losses or seeds. Compare original runtime JSON before making stronger claims.
The existing local HTML mentoring guide remains under ignored `ops-docs/`.
