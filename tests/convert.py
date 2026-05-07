
import pandas as pd

df = pd.read_parquet("train-00000-of-00002.parquet")

print(df.columns)

with open("output.txt", "w", encoding="utf-8") as f:
    for row in df.iloc[:, 0]:
        f.write(str(row) + "\n")
