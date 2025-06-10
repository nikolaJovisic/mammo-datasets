from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset

class EmbedDataset(Dataset):
    def __init__(self, csv_path: str, images_path: str, transform=lambda x: x, additional=None):
        self.df = pd.read_csv(csv_path)
        self.images_path = images_path
        self.transform = transform
        self.additional = additional
        
        print(f'Embed dataset initialized with {len(self.df)} images.')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.df.iloc[idx]['png_path'])
        
        img = Image.open(img_path)
        img = self.transform(img)
        
        if self.additional is None:
            return img
        else:
            return img, *[self.df[idx][i] for i in self.additional]
