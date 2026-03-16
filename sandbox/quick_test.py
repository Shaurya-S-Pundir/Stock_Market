import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/ml_features")

files = list(DATA_PATH.glob("*.csv"))

print("\nFound datasets:")
for f in files:
    print(f.name)

#print("\n--- DATA VALIDATION REPORT ---\n")


df1 = pd.read_csv(files[-1])
df2 = pd.read_csv(files[-2])

#print((df1["target"] == df2["target"]).mean())
T1 = df1["target"]
T2 = df2["target"]
count = 0
for i in range(len(T1)):
    if T1[i]!=T2[i]:
        count+=1
        
else: print("Chud gaye guru")
print(count)