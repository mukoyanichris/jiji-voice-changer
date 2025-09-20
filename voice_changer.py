#!/usr/bin/env python3
"""
JijiMinds Voice — real-time voice changer (LOCAL DSP, no network)
- Mic -> High-pass -> (optional) WebRTC NS/AGC -> Rubber Band pitch shift (formant-preserving, HQ) -> Limiter -> Output
- Linux: route this app's Playback to your VirtualMic sink (pavucontrol/qpwgraph)
- Windows (dev/test only): select your virtual cable as output

Presets:
  normal  = clean mic (HPF + optional NS/AGC + limiter)
  female  = +5 semitones (formant-preserving, HQ)
  accent  = -2 semitones (formant-preserving, HQ)

Runtime switching (stdin): type `n`, `f`, or `a` + Enter

Deps:
  sudo apt install rubberband-cli librubberband-dev
  pip install sounddevice numpy scipy pyrubberband
  # optional (works best on Py<=3.11; auto-skip on Py3.12): pip install webrtc-audio-processing
"""

import os
import sys
import time
import queue
import argparse
import threading
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter

# ---- Rubber Band (formant-preserving pitch shift) ----
try:
    import pyrubberband as prb
except Exception as e:
    print("❌ Missing pyrubberband. Install:\n"
          "  sudo apt install librubberband-dev rubberband-cli\n"
          "  pip install pyrubberband")
    raise

# ---- OPTIONAL: WebRTC NS/AGC (graceful fallback on Py 3.12) ----
USE_WEBRTC = True
try:
    from webrtc_audio_processing import AudioProcessing, NsConfig, AgcConfig
    _ap = AudioProcessing(
        enable_ns=True, ns_level=NsConfig.Level.HIGH,
        enable_agc=True, agc_mode=AgcConfig.Mode.FIXED_DIGITAL,
        enable_hpf=True
    )
    def denoise_agc(x: np.ndarray, sr: int) -> np.ndarray:
        xi = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        yi = _ap.process_stream(xi, sample_rate_hz=sr, num_channels=1)
        return (yi.astype(np.float32) / 32768.0)
except Exception as e:
    USE_WEBRTC = False
    print("⚠️  WebRTC NS/AGC not available on this Python/env (", e, "). Running without it.")
    def denoise_agc(x: np.ndarray, sr: int) -> np.ndarray:
        return x  # no-op fallback

# -------- Config --------
SR        = 48000     # PipeWire default
IN_HOP    = 1024      # callback block size (raise to 2048 if XRUNs; lower for latency)
PROC_LEN  = 6144      # worker chunk for Rubber Band (4096–6144–8192 = cleaner)
CHANNELS  = 1
INPUT_HINTS  = ["analog", "mic"]
OUTPUT_HINTS = ["pipewire", "default"]

APP_NAME = "JijiMinds Voice"
os.environ.setdefault("PULSE_PROP", f"application.name={APP_NAME}")  # label in pavucontrol

# -------- CLI --------
parser = argparse.ArgumentParser(description="JijiMinds Voice - local real-time voice changer")
parser.add_argument("--preset", choices=["normal", "female", "accent"], default="female", help="Starting preset")
args = parser.parse_args()

_current_preset = args.preset
_preset_lock = threading.Lock()
def set_preset(name: str):
    global _current_preset
    with _preset_lock:
        _current_preset = name
def get_preset() -> str:
    with _preset_lock:
        return _current_preset

# -------- Device helpers --------
def list_devices():
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        print(f"{i:2d} {d['name']} ({d['max_input_channels']} in, {d['max_output_channels']} out)")
    return devs
def pick_device(hints, kind):
    hints = [h.lower() for h in hints]
    for i, d in enumerate(sd.query_devices()):
        nm = d["name"].lower()
        if any(h in nm for h in hints):
            if kind == "input" and d["max_input_channels"] > 0: return i
            if kind == "output" and d["max_output_channels"] > 0: return i
    return None
def first_available(kind):
    for i, d in enumerate(sd.query_devices()):
        if kind == "input" and d["max_input_channels"] > 0: return i
        if kind == "output" and d["max_output_channels"] > 0: return i
    return None

# -------- Filters & limiter --------
def hp_filter(x: np.ndarray, sr: int, fc: float = 80.0) -> np.ndarray:
    b, a = butter(2, fc/(sr/2), btype="high")
    return lfilter(b, a, x)
def limiter(x: np.ndarray, th: float = 0.95) -> np.ndarray:
    return np.clip(x, -th, th)

# -------- Preset processing (Rubber Band HQ with formant preservation) --------
RB_ARGS = {"--formant": True, "--pitch-hq": True, "--crisp": 3}  # cleaner, less warble

def process_chain(x: np.ndarray, sr: int) -> np.ndarray:
    # Front-end cleanup
    x = hp_filter(x, sr)
    x = denoise_agc(x, sr)  # no-op on Py3.12 unless you installed webrtc-audio-processing

    p = get_preset()
    try:
        if p == "female":
            y = prb.pitch_shift(x, sr, n_steps=+5, rbargs=RB_ARGS)
        elif p == "accent":
            y = prb.pitch_shift(x, sr, n_steps=-2, rbargs=RB_ARGS)
        else:
            y = x
    except Exception as e:
        print("[RubberBand ERR]", e)
        # safe fallback without extra rbargs
        if p == "female":
            y = prb.pitch_shift(x, sr, n_steps=+5)
        elif p == "accent":
            y = prb.pitch_shift(x, sr, n_steps=-2)
        else:
            y = x

    return limiter(y)

# -------- Worker-threaded processing --------
in_q:  "queue.Queue[np.ndarray]" = queue.Queue(maxsize=40)
out_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=40)
_accum = np.zeros(0, dtype=np.float32)

def worker(sr: int):
    global _accum
    print(f"[worker] started (PROC_LEN={PROC_LEN}, WebRTC={'ON' if USE_WEBRTC else 'OFF'})")
    while True:
        x = in_q.get()
        if x is None:
            break
        _accum = np.concatenate([_accum, x])
        while len(_accum) >= PROC_LEN:
            chunk = _accum[:PROC_LEN].copy()
            _accum = _accum[PROC_LEN:]
            y = process_chain(chunk, sr)
            out_q.put(y)

# -------- Audio callback --------
def callback(indata, outdata, frames, time_info, status):
    if status:
        print("XRUN/status:", status, file=sys.stderr)

    x = indata[:, 0].astype(np.float32, copy=False)
    try:
        in_q.put_nowait(x)
    except queue.Full:
        pass  # drop if overflow

    try:
        y = out_q.get_nowait()
    except queue.Empty:
        y = x  # temporary passthrough

    if len(y) < frames:
        y = np.pad(y, (0, frames - len(y)))
    elif len(y) > frames:
        y = y[:frames]

    outdata[:, 0] = y
    if outdata.shape[1] > 1:
        outdata[:, 1:] = 0.0

# -------- Runtime preset switching --------
def stdin_listener():
    print("Type and press Enter:  n=normal, f=female, a=accent")
    while True:
        s = sys.stdin.readline().strip().lower()
        if   s in ("n","normal"): set_preset("normal"); print("Preset => normal")
        elif s in ("f","female"): set_preset("female"); print("Preset => female (+5)")
        elif s in ("a","accent"): set_preset("accent"); print("Preset => accent (-2)")
        else: print("Use: n / f / a")

# -------- Main --------
def main():
    print("Enumerating devices...")
    devs = list_devices()

    inp = pick_device(INPUT_HINTS, "input") or first_available("input")
    outp = pick_device(OUTPUT_HINTS, "output") or first_available("output")
    if inp is None:
        print("❌ No input (mic) device found."); sys.exit(1)
    if outp is None:
        print("❌ No output device found."); sys.exit(1)

    print(f"🎙  Input  -> {inp}: {devs[inp]['name']}")
    print(f"🔈 Output -> {outp}: {devs[outp]['name']}")
    print("\nLinux routing tip:\n"
          "  In pavucontrol (Playback), move 'JijiMinds Voice' to your 'VirtualMic' sink.\n"
          "  In Zoom, set Microphone = 'Monitor of VirtualMic' (or remapped 'JijiMic').\n")

    sd.default.samplerate = SR
    sd.default.blocksize  = IN_HOP
    sd.default.channels   = CHANNELS
    sd.default.device     = (inp, outp)

    threading.Thread(target=worker, args=(SR,), daemon=True).start()
    threading.Thread(target=stdin_listener, daemon=True).start()

    print("Move latest stream to VirtualMic (CLI):")
    print("  pactl move-sink-input $(pactl list short sink-inputs | tail -n1 | awk '{print $1}') VirtualMic\n")

    print(f"Starting… current preset = {get_preset()}  (Ctrl+C to stop)")
    try:
        with sd.Stream(callback=callback):
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        try: in_q.put_nowait(None)
        except Exception: pass
        print("\nBye!")

if __name__ == "__main__":
    main()
