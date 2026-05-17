#!/usr/bin/env bash
# Run this setup script with: source ./setup.sh

if [ ! -d ".venv" ]; then
    python3 -m venv .venv || return 1
fi

source .venv/bin/activate || return 1
python -m pip install -r requirements.txt || return 1
