#!/usr/bin/env bash

# UTF-8 environment variables (consistent with run.sh and run.bat)
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Install project dependencies into the current Python environment (no virtualenv).
# Make sure you have activated your desired conda/pyenv environment before running this.
python -m pip install -r requirements.txt

echo "Dependencies installed into current environment."
