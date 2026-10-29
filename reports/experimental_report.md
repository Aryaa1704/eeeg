# EEG Motor Imagery Classification Experimental Report

## Completed experiment

Nine BCI Competition IV Dataset 2a subjects were analysed using four classes, 22 EEG channels, 250 Hz sampling, Day-1 fitting, and Day-2 held-out evaluation. EOG channels and trials marked with event 1023 were excluded. The strict classifier was four-component CSP followed by one-vs-one shrinkage LDA. The paper-compatible filter was third-order zero-phase SOS Butterworth 0.5--30 Hz; 8--30 Hz was a separate experiment.

The motor-imagery baseline trained on 3.0--6.0 s. The visual-information-containing condition selected a 3-second window starting from 2.0 to 3.5 s in 0.1-second increments using three-fold CV on Day 1 only, and both conditions were tested on Day-2 3.0--6.0 s.

## Held-out results

Visual-information-containing training changed mean held-out accuracy by -1.18 percentage points in the 0.5--30 Hz experiment (Wilcoxon signed-rank statistic 9.0, two-sided p=0.1289) and -5.35 percentage points in the 8--30 Hz experiment (statistic 4.0, p=0.0547). Thus neither group result supports an across-subject improvement at alpha=0.05 under this protocol.

Subject 8 improved by 15.87 pp in the 0.5--30 Hz condition but declined by 15.87 pp in the 8--30 Hz condition. Effects are therefore subject- and frequency-band-dependent; they are not evidence that visual information improves practical motor-imagery decoding.

## Temporal windows

All 16 training-CV candidate windows per subject are saved in the temporal-window result files. For 0.5--30 Hz, selected starts for subjects 1--9 were 2.1, 3.2, 2.5, 2.7, 2.0, 2.0, 2.8, 2.4, and 2.0 s. For 8--30 Hz they were 2.4, 3.1, 2.4, 2.8, 2.8, 2.4, 3.0, 2.5, and 2.0 s. Held-out labels never selected a window.

## Comparison with the supplied paper

The supplied IEEE Access paper investigates how visual-presentation and rest-period data affect motor-imagery classification and cautions that visual information may inflate experimental performance without improving practical BCI relevance. Its workflow specifies 0.5--30 Hz preprocessing, four CSP features, one-vs-one LDA, 3-second windows, and Day-1/Day-2 evaluation; this strict classical implementation follows those elements. It does not reproduce the paper's SVM or CNN variants.

The supplied workflow records the paper's CSP-LDA statistical-comparison p-value as 0.001. That is a published reference value, not reproduced here. Differences may arise from artifact exclusions, exact window selection, library versions, labels, and the training-only window selection used here.

## Reproducibility outputs

Strict source outputs: `results_paper_temporal_retry` and `results_8_30_temporal`. Combined tables and five plots are in `results_final`. Configuration, per-class metrics, confusion matrices, selected windows, and statistical outputs are machine-readable in those folders.
