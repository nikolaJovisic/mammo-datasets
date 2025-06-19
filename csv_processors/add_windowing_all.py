import pandas as pd
import numpy as np
import sys
import os
import uuid
import concurrent.futures
import multiprocessing
import time

sys.path.append('..')

from mammo_dataset import MammoDataset, DatasetEnum, Split, ReturnMode
from utils.preprocess import negate_if_should
from windowing.calculate import calculate_a_b

def add_columns_to_csv(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['min_val', 'window_a', 'window_b', 'max_val']:
        if col not in df.columns:
            df[col] = None
    df.to_csv(csv_path, index=False)

def process_item(i):
    print(f"processing {i} started")
    img = dataset[i]
    img = negate_if_should(img)
    a, b = calculate_a_b(img)
    print(f"processing {i} finished")
    def to_int_or_nan(x):
        return int(x) if pd.notna(x) else pd.NA
    ret_val = i, to_int_or_nan(img.min()), to_int_or_nan(a), to_int_or_nan(b), to_int_or_nan(img.max())
    return ret_val

def writer_process(queue, csv_path, stop_token):
    while True:
        batch = []
        while not queue.empty():
            batch.append(queue.get())
        if batch:
            df = pd.DataFrame(batch, columns=['index', 'min_val', 'window_a', 'window_b', 'max_val'])
            df.set_index('index', inplace=True)
            df = df.astype({
                'min_val': 'Int64',
                'window_a': 'Int64',
                'window_b': 'Int64',
                'max_val': 'Int64'
            })

            original_df = pd.read_csv(csv_path, dtype={
                'min_val': 'Int64',
                'window_a': 'Int64',
                'window_b': 'Int64',
                'max_val': 'Int64'
            })

            cols_to_update = ['min_val', 'window_a', 'window_b', 'max_val']
            rows = df.index.values.astype(int)
            original_df.iloc[rows, original_df.columns.get_indexer(cols_to_update)] = df[cols_to_update].values
            original_df.to_csv(csv_path, index=False)

        if stop_token.is_set() and queue.empty():
            break
        time.sleep(1)



dataset_loaders = [
    lambda: MammoDataset(DatasetEnum.VINDR, Split.ALL, return_mode=ReturnMode.RAW_NUMPY),
    lambda: MammoDataset(DatasetEnum.RSNA, Split.ALL, return_mode=ReturnMode.RAW_NUMPY),
    lambda: MammoDataset(DatasetEnum.CSAW, Split.ALL, return_mode=ReturnMode.RAW_NUMPY),
    lambda: MammoDataset(DatasetEnum.EMBED, Split.TRAIN, return_mode=ReturnMode.RAW_NUMPY),
    lambda: MammoDataset(DatasetEnum.EMBED, Split.VALID, return_mode=ReturnMode.RAW_NUMPY),
    lambda: MammoDataset(DatasetEnum.EMBED, Split.TEST, return_mode=ReturnMode.RAW_NUMPY)
]

for load_dataset in dataset_loaders:
    dataset = load_dataset()
    add_columns_to_csv(dataset.csv_path)
    df = pd.read_csv(dataset.csv_path)
    missing_indices = df[df['min_val'].isna()].index
    
    print(f"Calculating windows for {len(missing_indices)} images from {dataset.ds_spec.name} dataset.")

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    stop_token = multiprocessing.Event()

    writer = multiprocessing.Process(target=writer_process, args=(queue, dataset.csv_path, stop_token))
    writer.start()

    def arg_generator():
        for i in missing_indices:
            yield i


    batch_size = 128
    batch = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=128) as executor:
        for result in executor.map(process_item, arg_generator(), chunksize=1):
            print(result)
            batch.append(result)
            if len(batch) >= batch_size:
                print(f"Writing batch of {len(batch)} rows.")
                for item in batch:
                    queue.put(item)
                batch.clear()

        if batch:
            for item in batch:
                queue.put(item)

    stop_token.set()
    writer.join()
