import os
import pandas as pd
from torch.utils.data import Dataset
from pydicom import dcmread

class VinDrDataset(Dataset):
    def __init__(self, csv_path: str, images_path: str, transform=lambda x: x, additional=None):
        self.df = pd.read_csv(csv_path)
        self.images_path = images_path
        self.transform = transform
        self.additional = additional
        
        print(f'VinDr dataset initialized with {len(self.df)} images.')
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_id = self.df.loc[idx, "study_id"]
        image_id = self.df.loc[idx, "image_id"]
        image_path = os.path.join(
            self.images_path, str(study_id), str(image_id) + ".dicom"
        )
        
        img = dcmread(image_path).pixel_array
        img = self.transform(img)

        if self.additional is None:
            return img
        else:
            return img, *[self.df[idx][i] for i in self.additional]

        