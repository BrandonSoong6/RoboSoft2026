import pandas as pd

df = pd.read_csv("canfield_lookup_0p4deg.csv.gz")
df[["q1_deg","q2_deg","q3_deg"]] = df[["q1_deg","q2_deg","q3_deg"]].round(3)
df_unique = df.drop_duplicates(subset=["q1_deg","q2_deg","q3_deg"])
df_unique.to_csv("canfield_lookup_0p4deg_unique.csv", index=False)
print("原始行数:", len(df))
print("去重后行数:", len(df_unique))