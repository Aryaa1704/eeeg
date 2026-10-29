from __future__ import annotations
import argparse
from pathlib import Path
from .config import ExperimentConfig
from .experiment import run
def main():
    p=argparse.ArgumentParser(); p.add_argument("data",type=Path); p.add_argument("--output",type=Path,default=Path("results")); p.add_argument("--band",choices=["paper","8-30"],default="paper"); a=p.parse_args()
    run(a.data,a.output,ExperimentConfig(band=a.band))

if __name__ == "__main__":
    main()
