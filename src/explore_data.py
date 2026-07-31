from datasets import load_dataset
import random

ds = load_dataset("DTU54DL/common-accent", split="train")

target_accents = [
    "India and South Asia (India, Pakistan, Sri Lanka)",
    "German English,Non native speaker",
    "Southern African (South Africa, Zimbabwe, Namibia)",
    "Filipino",
    "Singaporean English",
    "Hong Kong English",
]

filtered = ds.filter(lambda accent: accent in target_accents, input_columns=["accent"])

# Get just the accent labels (lightweight — doesn't touch audio)
accent_labels = filtered["accent"]

# Group row indices by accent
random.seed(42)
indices_by_accent = {a: [] for a in target_accents}
for i, accent in enumerate(accent_labels):
    indices_by_accent[accent].append(i)

# Sample up to 100 indices per accent
sampled_indices = []
for accent, idxs in indices_by_accent.items():
    random.shuffle(idxs)
    take = idxs[:100]
    sampled_indices.extend(take)
    print(f"{accent}: sampled {len(take)} of {len(idxs)}")

# Build the final balanced subset (still lazy — audio not decoded yet)
sample_ds = filtered.select(sampled_indices)
print(sample_ds)

# Save it to disk so run_inference.py can load it directly without redoing this
sample_ds.save_to_disk("data/accent_sample")