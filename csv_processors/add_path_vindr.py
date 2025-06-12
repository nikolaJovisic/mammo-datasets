import pandas as pd

df = pd.read_csv('vindr_breast_level.csv')
df['dcm_path'] = df['study_id'].astype(str) + '/' + df['image_id'].astype(str) + '.dicom'
df.to_csv('vindr_breast_level_a.csv', index=False)
