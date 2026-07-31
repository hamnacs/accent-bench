import pandas as pd

df = pd.read_csv("results/codeswitch_small_results.csv")
pd.set_option("display.max_colwidth", None)

for i, row in df.head(10).iterrows():
    wer_val = row['wer']
    wer_str = f"{wer_val:.2f}" if pd.notna(wer_val) else "N/A"
    print(f"\n--- {row['utt_id']} (lang={row['detected_language']}, WER={wer_str}) ---")
    print(f"Ground truth: {row['ground_truth']}")
    print(f"Prediction:   {row['prediction']}")