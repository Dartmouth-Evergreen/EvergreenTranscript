# transcriptionEvergreen
## Pipeline that uses whisperAI to transcribe the focus group meetings 
The bash script does: 
1. Covert audio to 16 kHz mono WAV
2. Runs whisper.cpp to generate plain text scripts
3. Runs Pyannote-Whisper to add speaker labels to each segment

You end up with, for every `audio/<name>.WAV:`

`audio/<name>.txt`

`audio/<name>_diarized.txt`

## Instructions for use
1. Clone the repo
```
#!/bin/bash
git clone https://github.com/your-org/whisper-transcription.git
cd whisper-transcription
```
2. Build the whisper.cpp
```
#!/bin/bash
cd whisper.cpp
make
./models/download-ggml-model.sh large-v1
cd ..
```

3. Install the python dependencies
```
#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install pywhispercpp pyannote-audio pyannote-whisper
```

### Usage
1. Add the audio files into audio directory (wav/WAV files through git cp)
2. Run the pipeline
```
#!/bin/bash
chmod +x run_pipeline.sh
./run_pipeline.sh en
```

### Output
Now under audio directory each WAV file should have a corresponding txt file
