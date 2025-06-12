#!/usr/bin/env bash
set -euo pipefail

LANG=${1:-en}
AUDIO_DIR="$(pwd)/audio"
MODEL_DIR="$(pwd)/whisper.cpp/models"
PYANNOTE_CFG="$(pwd)/pyannote-whisper/config.yaml"

echo "Starting full pipeline for files in ${AUDIO_DIR}…"

for src in "${AUDIO_DIR}"/*.WAV "${AUDIO_DIR}"/*.wav; do
  [ -f "$src" ] || continue
  base=$(basename "$src" | sed -E 's/\.(wav|WAV)$//')
  wav="${AUDIO_DIR}/${base}.wav"
  txt="${AUDIO_DIR}/${base}.txt"
  diarized="${AUDIO_DIR}/${base}_diarized.txt"

  echo; echo "=== Processing: $base ==="

  # 1) Convert to 16 kHz mono WAV
  ffmpeg -y -i "$src" -ar 16000 -ac 1 "$wav" < /dev/null

  # 2) Whisper transcription (no timestamps)
  ./whisper.cpp/build/bin/whisper-cli \
    -m "${MODEL_DIR}/ggml-large-v1.bin" \
    -f "$wav" \
    -l "$LANG" \
    --no-timestamps \
    --output-txt \
    --output-file "${AUDIO_DIR}/${base}"

  # 3) Diarization + merge
  python3 <<EOF
import sys
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
from pywhispercpp.model import Model

wav = r"$wav"
lang = r"$LANG"
model_dir = r"$MODEL_DIR"
cfg_file = r"$PYANNOTE_CFG"
out_file = r"$diarized"

# 1) load diarization pipeline (uncommented!)
pipeline = Pipeline.from_pretrained(cfg_file)

# 2) load Whisper model (with word timestamps)
wh = Model("large-v1", model_dir, n_threads=4)

# 3) run ASR with timestamps
asr = wh.transcribe(wav, language=lang, word_timestamps=True)

# 4) format for pyannote-whisper
res = {"segments": []}
for s in asr:
    res["segments"].append({
        "start": s.t0 / 100,
        "end":   s.t1 / 100,
        "text":  s.text
    })

# 5) run speaker diarization
diar = pipeline(wav)

# 6) merge transcripts & speaker turns
final = diarize_text(res, diar)

# 7) write out
with open(out_file, "w") as f:
    for seg, spk, txt in final:
        f.write(f"{seg.start:.2f}-{seg.end:.2f} {spk} {txt}\n")
EOF

  echo "→ Finished $base (transcript: $txt, diarized: $diarized)"
done

echo; echo "All done."

