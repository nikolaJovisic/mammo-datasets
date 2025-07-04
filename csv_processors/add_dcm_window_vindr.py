import pandas as pd
import pydicom
import yaml
from pathlib import Path

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

vindr_csv_path = config['vindr']['csv_path']
vindr_img_root = config['vindr']['images_path']

df = pd.read_csv(vindr_csv_path)
df['dcm_path'] = df['study_id'].astype(str) + '/' + df['image_id'].astype(str) + '.dicom'
df['dcm_path'] = df['dcm_path'].apply(lambda x: str(Path(vindr_img_root) / x))

def compute_window_bounds(dcm):
    wc = dcm.WindowCenter
    ww = dcm.WindowWidth
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = wc[0]
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = ww[0]
    a = wc - ww / 2
    b = wc + ww / 2
    return a, b

a_list = []
b_list = []

for path in df['dcm_path']:
    try:
        dcm = pydicom.dcmread(path)
        a, b = compute_window_bounds(dcm)
    except Exception:
        a, b = None, None
    a_list.append(a)
    b_list.append(b)

df['dcm_window_a'] = a_list
df['dcm_window_b'] = b_list
df.to_csv('vindr_breast_level_with_window.csv', index=False)
