# src/load_code_switch_data.py
import os
import random
import soundfile as sf
import numpy as np

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
                utt_id, text = parts
                texts[utt_id] = text
            elif len(parts) == 1:
                texts[parts[0]] = ""
    return texts

if __name__ == "__main__":
    wav_scp = load_wav_scp()
    segments = load_segments()
    texts = load_text()

    print(f"Recordings in wav.scp: {len(wav_scp)}")
    print(f"Segments: {len(segments)}")
    print(f"Transcripts: {len(texts)}")

    # Show a few examples
    for utt_id, rec_id, start, end in segments[:3]:
        print(f"\nUtterance: {utt_id}")
        print(f"Recording: {rec_id}, {start}s - {end}s")
        print(f"Text: {texts.get(utt_id, '[MISSING]')}")