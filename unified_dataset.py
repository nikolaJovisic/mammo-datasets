from math import gcd
from functools import reduce
from omegaconf import OmegaConf
from torch.utils.data import Dataset

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

        return_modes = [getattr(ds, 'return_mode', None) for ds in datasets]
        if not all(mode == return_modes[0] for mode in return_modes):
            raise ValueError('All datasets must have the same return_mode.')
        
        self.return_mode = return_modes[0]
        
        print(f'Unified dataset initialized with {len(self.indicies)} rows.')

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.indicies[idx]
        return self.datasets[ds_idx][sample_idx]
    
    def to_dict(self):
        combined_dict = {'dataset': []}
        for ds in self.datasets:
            d = ds.to_dict()
            combined_dict['dataset'].append(d.get('dataset', {}))
        return combined_dict
