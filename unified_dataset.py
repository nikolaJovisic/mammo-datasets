from math import gcd
from functools import reduce
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from utils.preprocess_transform import PreprocessTransform

from datasets.embed import EmbedDataset
from datasets.vindr import VinDrDataset
from datasets.rsna import RSNADataset

def get_datasets(datasets=['embed', 'vindr'], cfg_path='config.yaml'):
    cfg = OmegaConf.load(cfg_path)
    
    constructor = {
        'embed': lambda: EmbedDataset(cfg.embed.csv_path, cfg.embed.images_path, PreprocessTransform()),
        'vindr': lambda: VinDrDataset(cfg.vindr.csv_path, cfg.vindr.images_path, PreprocessTransform()),
        'rsna': lambda: RSNADataset(cfg.rsna.csv_path, cfg.rsna.images_path, PreprocessTransform()),
    }
    
    return [constructor[dataset]() for dataset in datasets] 

def get_unified_dataset(datasets=['embed', 'vindr']):
    return UnifiedDataset(get_datasets(datasets))

def proportional_round_robin_indices(lengths):
    total = sum(lengths)
    proportions = [l / total for l in lengths]
    counters = [0] * len(lengths)
    progress = [0.0] * len(lengths)

    result = []

    for _ in range(total):
        for i in range(len(progress)):
            progress[i] += proportions[i]

        while True:
            idx = max(range(len(progress)), key=lambda i: progress[i])
            if counters[idx] < lengths[idx]:
                result.append((idx, counters[idx]))
                counters[idx] += 1
                progress[idx] -= 1.0
                break
            else:
                progress[idx] = 0 

    return result

class UnifiedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.indicies = proportional_round_robin_indices([len(ds) for ds in datasets])

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.indicies[idx]
        return self.datasets[ds_idx][sample_idx]
