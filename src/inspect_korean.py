import pandas as pd

p = "data/zeroth_korean/data/test-00000-of-00001.parquet"

df = pd.read_parquet(p)

print(df)
print()
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print()
print("First example:")
print(df.iloc[0])