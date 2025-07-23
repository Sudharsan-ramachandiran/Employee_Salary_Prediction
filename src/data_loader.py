import pandas as pd
import os

def extract_and_load(zip_file, extract_dir):
    csv_path = os.path.join(extract_dir, "DataScience_salaries_2025.csv")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
