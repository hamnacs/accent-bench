import pandas as pd

df = pd.read_csv("results/transliteration_failures.csv")
pd.set_option("display.max_colwidth", None)

for i, row in df.head(8).iterrows():
    print(f"\n--- {i} ---")
    print("GT:  ", row["ground_truth"])
    print("Pred:", row["prediction"])