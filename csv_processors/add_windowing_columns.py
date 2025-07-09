import os
import pandas as pd

folder_path = '../csvs'
columns_to_add = ['min_val', 'max_val', 'window_a', 'window_b']

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        missing_columns = [col for col in columns_to_add if col not in df.columns]
        if missing_columns:
            for col in missing_columns:
                df[col] = None
            df.to_csv(file_path, index=False)
