#!/usr/bin/env bash
set -e
sudo apt update
# Virtual sink (PipeWire/PulseAudio null sink)
pactl load-module module-null-sink media.class=Audio/Sink \
  sink_name=VirtualMic sink_properties=node.description=VirtualMic || true

# Rubber Band (HQ pitch shifting backend)
sudo apt install -y rubberband-cli librubberband-dev python3-venv

# Python deps
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "✔ Setup complete. To run:"
echo "source venv/bin/activate"
echo "PULSE_PROP='application.name=JijiMinds Voice' python voice_changer.py --preset female"
