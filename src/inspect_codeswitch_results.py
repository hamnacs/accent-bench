# src/inspect_codeswitch_results.py
import pandas as pd

df = pd.read_csv("results/codeswitch_results.csv")
pd.set_option("display.max_colwidth", None)

for i, row in df.head(10).iterrows():
    print(f"\n--- {row['utt_id']} (lang={row['detected_language']}, WER={row['wer']:.2f}) ---")
    print(f"Ground truth: {row['ground_truth']}")
    print(f"Prediction:   {row['prediction']}")