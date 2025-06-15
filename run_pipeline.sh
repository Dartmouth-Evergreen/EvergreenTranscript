#!/usr/bin/env bash
set -euo pipefail
set -x

# 0) Sandbox caches under project
export HOME="$(pwd)"
export HF_HOME="$HOME/.hf_cache"
export TORCH_HOME="$HOME/.torch_cache"
export XDG_CACHE_HOME="$HF_HOME"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# 1) HF token
export HUGGINGFACE_HUB_TOKEN=[your token]

# 2) Language
LANG=${1:-en}

# 3) Directories
AUDIO_DIR="$PWD/audio"
MODEL_DIR="$PWD/whisper.cpp/models"

echo "Starting full pipeline in $AUDIO_DIR…"

for src in "$AUDIO_DIR"/*.WAV "$AUDIO_DIR"/*.wav; do
  [ -f "$src" ] || continue
  base=$(basename "$src" | sed -E 's/\.(wav|WAV)$//')
  wav="$AUDIO_DIR/$base.wav"
  csv="$AUDIO_DIR/$base.csv"
  diar="$AUDIO_DIR/${base}_diarized.txt"

  echo; echo "=== $base ==="

  # 1) Resample
  ffmpeg -y -i "$src" -ar 16000 -ac 1 "$wav" < /dev/null

  # 2) ASR → CSV
  if [[ -s "$csv" ]]; then
    echo "→ CSV exists, skipping Whisper"
  else
    echo "→ Whisper → CSV on $base"
    ./whisper.cpp/build/bin/whisper-cli \
      --debug-mode \
      --print-progress \
      -m "$MODEL_DIR/ggml-large-v1.bin" \
      -f "$wav" \
      -l "$LANG" \
      -ocsv \
      -of "$AUDIO_DIR/$base"
  fi

  # 3) Diarization + merge
  if [[ -s "$diar" ]]; then
    echo "→ Diarized file exists, skipping"
  else
    echo "→ Diarization & merge on $base"
    python3 <<EOF
import os, sys, csv, torch
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text

torch.hub.set_dir(os.environ["TORCH_HOME"])
wav      = r"$wav"
csv_file = r"$csv"
out_file = r"$diar"

# 3.1) load ASR segments
segments = []
with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            s, e = float(row['start']), float(row['end'])
        except:
            continue
        segments.append({"start": s, "end": e, "text": row['text']})

print(f"[DEBUG] ASR segments: {len(segments)}", file=sys.stderr)

# 3.2) load diarization model
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ["HUGGINGFACE_HUB_TOKEN"]
)
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

# 3.3) apply to WAV
diar = pipeline(wav)
print(f"[DEBUG] Speaker turns: {len(diar.get_timeline())}", file=sys.stderr)

# 3.4) merge
merged = diarize_text({"segments": segments}, diar)
print(f"[DEBUG] Merged segments: {len(merged)}", file=sys.stderr)

# 3.5) write
with open(out_file, "w") as out:
    for seg, spk, txt in merged:
        out.write(f"{seg.start:.2f}-{seg.end:.2f} {spk} {txt}\n")
print(f"[DEBUG] Wrote {out_file!r}", file=sys.stderr)
EOF
  fi

  echo "→ Done $base → (csv=$csv; diarized=$diar)"
done

echo; echo "All done."

