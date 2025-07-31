import pandas as pd
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(ROOT_DIR)


def merge(df1, df2):
  df1 = df1.copy()
  df2 = df2.copy()
  print(len(df1), len(df2))

  df2['dialogue'] = df1['dialogue']
  

  # df1 = pd.concat([df1, df2], ignore_index=True)
  return df2


if __name__ == "__main__":
  df1 = pd.read_csv(os.path.join(ROOT_DIR, "data", "organized_dialogues.csv"))
  df2 = pd.read_csv(os.path.join(ROOT_DIR, "data", "train.csv"))
  df = merge(df1, df2)
  df.to_csv(os.path.join(ROOT_DIR, "data", "train_organized.csv"), index=False)