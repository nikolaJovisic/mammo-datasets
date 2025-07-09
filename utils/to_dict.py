import yaml
from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

def dataset_to_dict(dataset, top_element='dataset'):
    dataset_info = {}

    dataset_info['constructor_args'] = {
        'dataset': dataset.ds_spec.name,
        'return_mode': dataset.return_mode.name,
        'convert_to': dataset.format_transform.convert_to.name,
        'labels': dataset.labels,
        'clahe': dataset.preprocess_transform.clahe,
        'aspect_ratio': dataset.preprocess_transform.aspect_ratio,
        'resize': list(dataset.preprocess_transform.resize) if dataset.preprocess_transform.resize is not None else None,
        'tile_size': list(dataset.tile_size) if dataset.tile_size is not None else None,
        'tile_overlap': dataset.tile_overlap,
        'final_resize': list(dataset.format_transform.resize) if dataset.format_transform.resize is not None else None,
        'return_index': dataset.return_index,
        'cfg_path': str(dataset.cfg_path),
        'csv_path': str(dataset.csv_path),
        'images_path': str(dataset.images_path),
        'limit': str(dataset.limit),
        'read_window' : str(dataset.read_window)
    }

    dataset_info['dataset_stats'] = {
        'length': len(dataset),
        'label_distribution': dataset.df[dataset.ds_spec.label_col].value_counts().to_dict(),
    }

    return {
        top_element: dataset_info
    }