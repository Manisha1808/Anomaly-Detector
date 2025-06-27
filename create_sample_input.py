# create_sample_input.py

import pandas as pd
df = pd.read_csv("data/cleaned_kdd.csv")
df = df.dropna(subset=["target"])
df = df.drop("target", axis=1)
df.sample(5).to_csv("data/new_input.csv", index=False)
