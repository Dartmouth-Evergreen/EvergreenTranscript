#!/usr/bin/env python3
import sys
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
from pywhispercpp.model import Model

if len(sys.argv) != 2:
    print("Usage: diarize.py <wav-file>", file=sys.stderr)
    sys.exit(1)

wav = sys.argv[1]
# Paths (adjust if your layout is different):
MODEL_DIR = "../whisper.cpp/models"
CFG      = "../pyannote-whisper/config.yaml"

# 1) load diarization pipeline
pipeline = Pipeline.from_pretrained(CFG)

# 2) load Whisper model (with timestamps)
wh = Model("large-v1", MODEL_DIR, n_threads=4)
asr_segments = wh.transcribe(wav, language="en", word_timestamps=True)

# 3) reformat for pyannote-whisper
result = {"segments": []}
for s in asr_segments:
    result["segments"].append({
        "start": s.t0 / 100,
        "end":   s.t1 / 100,
        "text":  s.text
    })

# 4) run speaker diarization
diar = pipeline(wav)

# 5) merge & print
final = diarize_text(result, diar)
for seg, spk, txt in final:
    print(f"{seg.start:.2f}-{seg.end:.2f} {spk} {txt}")

