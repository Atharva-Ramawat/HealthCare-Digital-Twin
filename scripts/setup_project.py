#!/usr/bin/env python
"""
setup_project.py
----------------
One-time project setup script.
Creates required directories, validates config, checks environment.
"""
import os, sys, pathlib

def main():
    root = pathlib.Path(__file__).parent.parent
    required_dirs = [
        root / "data" / "raw" / "mimic_iv",
        root / "data" / "raw" / "mimic_waveform",
        root / "data" / "raw" / "vitaldb",
        root / "data" / "raw" / "eicu",
        root / "data" / "interim",
        root / "data" / "processed",
        root / "data" / "features",
        root / "data" / "synthetic",
        root / "models" / "checkpoints",
        root / "models" / "registry",
        root / "logs",
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("Project directories initialised.")
    # TODO: validate settings.yaml, check Python version, check venv

if __name__ == "__main__":
    main()
