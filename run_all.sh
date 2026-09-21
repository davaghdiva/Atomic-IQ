#!/usr/bin/env bash
set -euo pipefail
python3 run_calibration.py
python3 run_primary_gmv.py
pytest -q
