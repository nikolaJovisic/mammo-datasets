import os
import shutil
import csv
import cv2
from concurrent.futures import ProcessPoolExecutor
import sys

sys.path.append('../..')

from mammo_dataset import MammoDataset, DatasetEnum, Split, ReturnMode
from utils.preprocess import negate_if_should
from windowing.calculate import calculate_a_b
from windowing.apply import window

OUTPUT_FOLDER = 'data'

def process_image(args):
    image, name, i = args
    
    image = negate_if_should(image)
    
    gcs_windowed = window(image, image.min(), image.max())
    a, b = calculate_a_b(image)
    grail_windowed = window(image, a, b)

    image_base = f"{OUTPUT_FOLDER}/{name}/{i}"
    cv2.imwrite(f"{image_base}_gcs.png", gcs_windowed)
    cv2.imwrite(f"{image_base}_grail.png", grail_windowed)

    return (name, f"{i}", image.min(), a, b, image.max())

def save_metadata_row(row):
    name, *rest = row
    csv_path = f"{OUTPUT_FOLDER}/{name}.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(['name', 'min', 'a', 'b', 'max'])
        writer.writerow([rest[0]] + rest[1:])

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

datasets = [
    (MammoDataset(DatasetEnum.VINDR, Split.TRAIN, labels=['BI-RADS 1'], return_mode=ReturnMode.RAW_NUMPY)[:50], "vindr_healthy"),
    (MammoDataset(DatasetEnum.VINDR, Split.TRAIN, labels=['BI-RADS 4', 'BI-RADS 5', 'BI-RADS 6'], return_mode=ReturnMode.RAW_NUMPY)[:50], "vindr_diseased"),
    (MammoDataset(DatasetEnum.RSNA, Split.TRAIN, labels=1, return_mode=ReturnMode.RAW_NUMPY)[:50], "rsna_diseased"),
    (MammoDataset(DatasetEnum.RSNA, Split.TRAIN, labels=0, return_mode=ReturnMode.RAW_NUMPY)[:50], "rsna_healthy"),
    (MammoDataset(DatasetEnum.CSAW, Split.TRAIN, labels=1, return_mode=ReturnMode.RAW_NUMPY)[:50], "csaw_diseased"),
    (MammoDataset(DatasetEnum.CSAW, Split.TRAIN, labels=0, return_mode=ReturnMode.RAW_NUMPY)[:50], "csaw_healthy"),
    (MammoDataset(DatasetEnum.EMBED, Split.TRAIN, labels=[4, 5, 6], return_mode=ReturnMode.RAW_NUMPY)[:50], "embed_diseased"),
    (MammoDataset(DatasetEnum.EMBED, Split.TRAIN, labels=1, return_mode=ReturnMode.RAW_NUMPY)[:50], "embed_healthy")
]

for _, name in datasets:
    shutil.rmtree(f"{OUTPUT_FOLDER}/{name}", ignore_errors=True)
    os.mkdir(f"{OUTPUT_FOLDER}/{name}")
    csv_file = f"{OUTPUT_FOLDER}/{name}.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)

tasks = []
for dataset, name in datasets:
    for i, image in enumerate(dataset):
        tasks.append((image, name, i))

with ProcessPoolExecutor(max_workers=128) as executor:
    for result in executor.map(process_image, tasks):
        save_metadata_row(result)
