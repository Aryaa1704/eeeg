from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
try:
    from mne.decoding import CSP
except ImportError:
    CSP = None

def filter_trials(x: np.ndarray, sfreq: float, band: tuple[float, float], order: int) -> np.ndarray:
    sos = butter(order, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)

def crop(x: np.ndarray, sfreq: float, window: tuple[float, float]) -> np.ndarray:
    start, end = (round(v * sfreq) for v in window)
    if start < 0 or end > x.shape[-1] or start >= end: raise ValueError(f"Window {window} outside epoch")
    return x[..., start:end]

def make_model(n_components: int) -> Pipeline:
    if CSP is None: raise RuntimeError("MNE is required for the CSP baseline; install .[research]")
    # CSP's one-vs-rest multiclass mode fits filters only during pipeline.fit.
    return Pipeline([("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False, component_order="mutual_info")),
                     ("scale", StandardScaler()), ("lda", OneVsOneClassifier(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")))])
