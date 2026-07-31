import pandas as pd
import re

df = pd.read_csv("results/codeswitch_small_results.csv")
df = df.dropna(subset=["prediction", "ground_truth"])

def extract_latin_words(text):
    # Words made of Latin letters (i.e., the English portion of code-switched text)
    return set(re.findall(r'[a-zA-Z]+', text.lower()))

def transliteration_failure(row):
    gt_english_words = extract_latin_words(row["ground_truth"])
    if not gt_english_words:
        return False  # no English words in ground truth, not applicable
    pred_english_words = extract_latin_words(row["prediction"])
    # If ground truth had English words but prediction has few/none in Latin script,
    # likely transliterated into Devanagari instead
    overlap = gt_english_words & pred_english_words
    return len(overlap) == 0

def likely_hallucination(row):
    gt_len = len(row["ground_truth"])
    pred_len = len(row["prediction"])
    if gt_len == 0:
        return False
    return pred_len > gt_len * 1.6  # prediction notably longer than source

df["has_english_in_gt"] = df["ground_truth"].apply(lambda x: len(extract_latin_words(x)) > 0)
df["transliteration_failure"] = df.apply(transliteration_failure, axis=1)
df["likely_hallucination"] = df.apply(likely_hallucination, axis=1)

applicable = df[df["has_english_in_gt"]]
print(f"Utterances containing English words in ground truth: {len(applicable)} / {len(df)}")
print(f"Of those, transliteration failures (no matching Latin-script words in prediction): "
      f"{applicable['transliteration_failure'].sum()} / {len(applicable)} "
      f"({applicable['transliteration_failure'].mean()*100:.1f}%)")

print(f"\nLikely hallucinations (prediction >1.6x longer than ground truth) across all {len(df)} utterances: "
      f"{df['likely_hallucination'].sum()} ({df['likely_hallucination'].mean()*100:.1f}%)")

# Save flagged examples for manual spot-check
flagged = applicable[applicable["transliteration_failure"]][["ground_truth", "prediction"]]
flagged.to_csv("results/transliteration_failures.csv", index=False)
print(f"\nSaved {len(flagged)} transliteration-failure examples to results/transliteration_failures.csv")