from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .config import EVENT_TO_LABEL

@dataclass
class Session:
    data: np.ndarray  # trials, EEG channels, samples; no EOG
    labels: np.ndarray
    sfreq: float

def _evaluation_labels(path: Path) -> np.ndarray:
    values = loadmat(path).get("classlabel")
    if values is None: raise ValueError(f"{path} does not contain classlabel")
    labels = np.asarray(values).squeeze().astype(int) - 1
    if not np.all(np.isin(labels, range(4))): raise ValueError(f"Invalid class labels in {path}")
    return labels

def load_gdf(path: Path, labels_path: Path | None = None, *, tmin: float = 0., tmax: float = 6.5) -> Session:
    """Epoch from official 768 trial starts; class events or authorised .mat labels identify targets."""
    try: import mne
    except ImportError as exc: raise RuntimeError("Install the research extra: pip install -e .[research]") from exc
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    # Dataset 2a GDF files label the three EOG traces as EEG in their metadata.
    # Exclude them by their documented channel names, rather than trusting type.
    eeg = mne.pick_types(raw.info, eeg=True, eog=False,
                         exclude=[name for name in raw.ch_names if name.upper().startswith("EOG")])
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    inverse = {v: int(k) for k, v in event_id.items() if str(k).isdigit()}
    decoded = [(sample, inverse.get(code)) for sample, _, code in events]
    starts = [i for i, (_, code) in enumerate(decoded) if code == 768]
    if not starts: raise ValueError(f"No official 768 trial starts in {path}")
    rejected = []
    for pos, start_index in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(decoded)
        rejected.append(any(code == 1023 for _, code in decoded[start_index:end]))
    if labels_path:
        targets = _evaluation_labels(labels_path)
        if len(targets) != len(starts): raise ValueError(f"{path}: {len(starts)} trials but {len(targets)} external labels")
    else:
        targets=[]
        for pos, start_index in enumerate(starts):
            end = starts[pos + 1] if pos + 1 < len(starts) else len(decoded)
            code = next((c for _, c in decoded[start_index:end] if c in EVENT_TO_LABEL), None)
            if code is None: raise ValueError(f"{path}: missing 769–772 class event for trial {pos + 1}; provide authorised .mat labels")
            targets.append(EVENT_TO_LABEL[code])
        targets=np.asarray(targets)
    clean = np.flatnonzero(~np.asarray(rejected))
    event_array = np.array([[decoded[starts[i]][0], 0, 768] for i in clean])
    epochs = mne.Epochs(raw, event_array, event_id={"trial_start": 768}, tmin=tmin, tmax=tmax,
                        picks=eeg, baseline=None, preload=True, reject_by_annotation=True, verbose="ERROR")
    return Session(epochs.get_data(), np.asarray(targets)[clean][epochs.selection], float(raw.info["sfreq"]))

def find_subject_sessions(root: Path, subject: int) -> tuple[Path, Path, Path]:
    return root / f"A{subject:02d}T.gdf", root / f"A{subject:02d}E.gdf", root / "true_labels" / f"A{subject:02d}E.mat"
