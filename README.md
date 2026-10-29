# EEG motor-imagery research pipeline

Reproducible, leakage-aware four-class analysis for BCI Competition IV Dataset 2a.
The primary benchmark trains on each subject's `A0?T.gdf` recording and evaluates
on its `A0?E.gdf` recording. It never substitutes a random trial split.

## Install and run

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e .[research]
eeg-mi data --output results
```

Put the official `A01T.gdf` … `A09T.gdf` and evaluation recordings
`A01E.gdf` … `A09E.gdf` in `data/`, and authorised `A01E.mat` … `A09E.mat`
class-label files in `data/true_labels/`. Evaluation labels are not embedded in
the original competition evaluation recordings. The command exits without
making up results when they are absent.

`--band paper` evaluates the paper-compatible 0.5–30 Hz preprocessing;
`--band 8-30` is a separate experiment. The default uses the transparent
multiclass CSP + LDA baseline. `--exploratory-test-window-search` records
test-scored windows but labels them exploratory.

The generated output schema is documented in `reports/experimental_report.md`.
