#!/bin/bash
VENV=$(find /var/app/venv -maxdepth 1 -type d | tail -1)
source $VENV/bin/activate
pip install scikit-learn==1.3.2 --force-reinstall --quiet
