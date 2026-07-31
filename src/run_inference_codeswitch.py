import os
import random
import soundfile as sf
import numpy as np
import whisper
from jiwer import wer
import pandas as pd
import tempfile
import time

BASE = "data/mucs_hindi_english/test"

def load_wav_scp():
    wav_paths = {}
    with open(f"{BASE}/transcripts/wav.scp", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                rec_id, path = parts
                wav_paths[rec_id] = path
    return wav_paths

def load_segments():
    segments = []
    with open(f"{BASE}/transcripts/segments", encoding="utf-8") as f:
        for line in f:
            utt_id, rec_id, start, end = line.strip().split()
            segments.append((utt_id, rec_id, float(start), float(end)))
    return segments

def load_text():
    texts = {}
    with open(f"{BASE}/transcripts/text", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]
            elif len(parts) == 1:
                texts[parts[0]] = ""
    return texts

wav_scp = load_wav_scp()
segments = load_segments()
texts = load_text()

# Sample 100 segments randomly, but keep only ones with real transcript text
random.seed(42)
valid_segments = [s for s in segments if texts.get(s[0], "").strip() != ""]
sample = random.sample(valid_segments, min(100, len(valid_segments)))

print(f"Sampled {len(sample)} code-switched utterances")

model = whisper.load_model("base")

# Cache loaded recordings so we don't re-read the same long WAV repeatedly
audio_cache = {}

results = []

for i, (utt_id, rec_id, start, end) in enumerate(sample):
    ground_truth = texts[utt_id]

    if rec_id not in audio_cache:
        # wav.scp path is relative to the extracted folder structure
        wav_filename = f"{rec_id}.wav"
        wav_path = os.path.join(BASE, wav_filename)
        if not os.path.exists(wav_path):
            print(f"Missing audio file: {wav_path}")
            continue
        audio_array, sr = sf.read(wav_path, dtype="float32")
        audio_cache[rec_id] = (audio_array, sr)

    audio_array, sr = audio_cache[rec_id]
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    clip = audio_array[start_sample:end_sample]

    if len(clip) == 0:
        continue

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, clip, sr)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, fp16=False)
        prediction = result["text"].strip()
        detected_lang = result.get("language", "unknown")
        error_rate = wer(ground_truth, prediction)
    except Exception as e:
        print(f"[{i+1}/{len(sample)}] Error: {e}")
        prediction = ""
        detected_lang = "error"
        error_rate = None
    finally:
        for attempt in range(5):
            try:
                os.remove(tmp_path)
                break
            except PermissionError:
                time.sleep(0.2)

    results.append({
        "utt_id": utt_id,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "detected_language": detected_lang,
        "wer": error_rate
    })

    if error_rate is not None:
        print(f"[{i+1}/{len(sample)}] lang={detected_lang:6s} | WER: {error_rate:.2f}")

    if (i + 1) % 25 == 0:
        pd.DataFrame(results).to_csv("results/codeswitch_progress.csv", index=False)

df = pd.DataFrame(results)
df.to_csv("results/codeswitch_results.csv", index=False)

print(f"\nDone. {df['wer'].notna().sum()} / {len(df)} successful")
print(f"\nMean WER: {df['wer'].mean():.3f}")
print(f"Median WER: {df['wer'].median():.3f}")
print("\nDetected language distribution:")
print(df["detected_language"].value_counts())