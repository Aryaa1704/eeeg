# EEG motor imagery research handoff

## Project location

`C:\Users\anshi\Documents\Codex\2026-10-29\github-plugin-github-openai-curated-remote`

Run all commands from that directory. The raw dataset has been copied locally and must not be put in Git:

```text
data\A01T.gdf ... A09T.gdf
data\A01E.gdf ... A09E.gdf
data\true_labels\A01E.mat ... A09E.mat
```

Reference materials remain in Downloads:

```text
C:\Users\anshi\Downloads\desc_2a.pdf
C:\Users\anshi\Downloads\Effects_of_Data_Including_Visual_Presentation_and_Rest_Time_on_Classification_of_Motor_Imagery_of_Using_Brain-Computer_Interface_Competition_Datasets.pdf
C:\Users\anshi\Downloads\BCI_MI_Visual_Presentation_Complete_Workflow.docx
```

## Completed work

The Python package is in `src\eeg_mi`. It reads Dataset 2a GDF files, removes `EOG-*` channels by name (they are incorrectly typed as EEG in the GDF metadata), anchors trials at event 768, maps 769--772 training labels, loads authorised evaluation labels from `.mat`, and excludes trials containing event 1023.

The strict paper-compatible CSP-LDA benchmark has completed for all 9 subjects. Its output is in `results_paper_strict\`. It uses:

- Day-1 `A0?T.gdf` fitting and Day-2 `A0?E.gdf` held-out evaluation;
- 22 EEG channels only;
- third-order zero-phase SOS Butterworth 0.5--30 Hz filtering;
- 4 CSP components;
- one-vs-one LDA;
- MI-only baseline 3.0--6.0 seconds;
- visual-information window search using 3-second windows from 2.0 through 3.5 seconds in 0.1-second steps, selected by 3-fold CV on the training session only.

Earlier folders `results\` and `results_8_30\` contain preliminary runs and must **not** be used for final claims: they used 8 CSP features and a wider exploratory window range.

## Current status

Both strict experiments are complete and verified with 9 completed subjects and no captured error output:

```text
results_paper_strict\    0.5--30 Hz paper-compatible benchmark
results_8_30_strict\     separate strict 8--30 Hz comparison
```

Temporal tables, corrected statistical output, Phase 4 figures, and the final report are complete. Combined artifacts are in `results_final\`, including five PNG figures and `combined_group_statistics.csv`; the final report is `reports\experimental_report.md`. No computation is currently running.

The paper-band temporal-window regeneration completed successfully (9 subjects; `results_paper_temporal_retry\temporal_window_results.csv` contains all 144 training-CV rows).

The matching strict 8--30 Hz temporal-window regeneration also completed successfully (9 subjects; `results_8_30_temporal\temporal_window_results.csv` contains all 144 training-CV rows).

```text
Output: results_8_30_temporal\
```

The first paper attempt (`paper_temporal`) stopped after a window-selection tuple bug; it is invalid and must be ignored. Both completed temporal runs persist all 16 training-CV window rows per subject and corrected Holm-adjusted subject-level p-values. The next active phase is generation of figures and the final report from these two temporal result directories. No EEG model process is currently running.

## Required next work

1. After the current strict 8--30 Hz run finishes, extend `choose_window` in `src\eeg_mi\experiment.py` so every training-CV candidate window is retained in `temporal_window_results.csv`. The current CSV is only a placeholder. If scoring every window on held-out test data, label it `exploratory=True`; do not present it as unbiased selection.

2. Fix Holm adjustment persistence in `experiment.py`: it is currently assigned to a temporary DataFrame (`tdf`) and is not written back to `tests`, so `adjusted_p_value_holm` is blank in `statistical_tests.csv`. Add the stated test metadata, effect size, and confidence interval where supportable.

3. Generate the requested publication-quality plots in each strict result directory: subject accuracy comparison, per-class comparison, time-vs-accuracy, width plot if secondary widths are run, heatmap, confusion matrices, distributions, and condition-difference plot. Never create plots from invented values.

4. Replace the initial placeholder `reports\experimental_report.md` with a report based only on strict result folders. Compare it with verified paper facts: Dataset 2a has 9 subjects, 22 EEG + 3 EOG, 250 Hz, 288 trials/session, cue starts at 2 s and MI ends at 6 s; the paper reports 0.5--30 Hz third-order preprocessing and a 3-second 2.0--3.5 s window search. The paper's reported CSP-LDA statistical comparison p-value is 0.001; cite it as published, not reproduced. Clearly label all project values as newly generated.

5. Run tests after modifications. The last local run passed `2 passed` before strict-run completion.

## Useful status commands

```powershell
Get-Content .\results_paper_strict\summary_statistics.json
Import-Csv .\results_paper_strict\subject_wise_accuracy.csv | Format-Table
Get-ChildItem .\results_paper_strict\confusion_matrices
```

## Important research constraints

- Do not use random trial-level train/test splits.
- Do not use EOG as classifier features.
- Do not claim a visual/no-visual human experiment: these are different training-window selections from the same trials.
- Do not fabricate accuracy, p-values, figures, or paper values.
- Do not overwrite strict results or raw GDF data.
