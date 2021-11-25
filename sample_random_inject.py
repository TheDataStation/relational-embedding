import pandas as pd 
import numpy as np 

num_noise_rows = 5
df = pd.read_csv("./data/sample/base.csv")
for i in range(num_noise_rows):
    df["features " + str(i)] = np.random.random(df["name"].shape)
print(df)
df.to_csv("./data/sample_frac/base.csv")

num_noise_col_2 = 5
df = pd.read_csv("./data/sample/trans.csv")
for i in range(num_noise_col_2):
    df["features " + str(i)] = np.random.random(df["item"].shape)
df.to_csv("./data/sample/trans.csv")