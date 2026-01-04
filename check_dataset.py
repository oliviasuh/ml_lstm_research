import pandas as pd

csv_path = "accident_dataset.csv"
dataset = pd.read_csv(csv_path)
print(f"Dataset columns: {dataset.columns}")
print(f"First few rows:\n{dataset.head()}")
