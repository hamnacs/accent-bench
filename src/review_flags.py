import pandas as pd

df = pd.read_csv("results/language_switch_examples.csv")
pd.set_option("display.max_colwidth", None)

for i, row in df.iterrows():
    print(f"\n--- {row['accent']} (WER: {row['wer']:.2f}) ---")
    print(f"Ground truth: {row['ground_truth']}")
    print(f"Prediction:   {row['prediction']}")