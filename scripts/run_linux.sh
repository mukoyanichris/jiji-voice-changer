#!/usr/bin/env bash
set -e
source venv/bin/activate
PULSE_PROP="application.name=JijiMinds Voice" \
python voice_changer.py --preset female
