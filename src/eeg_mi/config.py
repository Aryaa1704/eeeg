from __future__ import annotations
from dataclasses import dataclass, asdict

EVENT_TO_LABEL = {769: 0, 770: 1, 771: 2, 772: 3}
CLASS_NAMES = ["left_hand", "right_hand", "both_feet", "tongue"]

@dataclass(frozen=True)
class ExperimentConfig:
    sfreq: float = 250.0
    band: str = "paper"  # paper = 0.5–30 Hz; 8-30 is a distinct study
    filter_order: int = 3
    mi_window: tuple[float, float] = (3.0, 6.0)
    cue_window: tuple[float, float] = (2.0, 5.0)
    # Strict primary reproduction in the supplied workflow. Additional widths
    # are a separately labelled secondary analysis, not the paper benchmark.
    search_widths: tuple[float, ...] = (3.0,)
    search_start_range: tuple[float, float] = (2.0, 3.5)
    step_seconds: float = 0.1
    random_seed: int = 2023
    csp_components: int = 4
    def band_limits(self) -> tuple[float, float]: return (0.5, 30.0) if self.band == "paper" else (8.0, 30.0)
    def metadata(self) -> dict: return asdict(self) | {"band_limits_hz": self.band_limits(), "filter": "Butterworth, SOS, zero-phase sosfiltfilt; padding handled by SciPy"}
