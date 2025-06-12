import pandas as pd

df = pd.read_csv('rsna.csv')
df['dcm_path'] = df['patient_id'].astype(str) + '/' + df['image_id'].astype(str) + '.dcm'
df.to_csv('rsna_with_dcm_path.csv', index=False)