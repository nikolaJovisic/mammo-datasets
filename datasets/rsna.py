import os
import pandas as pd
from torch.utils.data import Dataset
from pydicom import dcmread

class RSNADataset(Dataset):
    def __init__(self, csv_path: str, images_path: str, transform=lambda x: x, additional=None):
        self.df = pd.read_csv(csv_path)
        self.images_path = images_path
        self.transform = transform
        self.additional = additional
        
        print(f'RSNA dataset initialized with {len(self.df)} images.')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = self.data.loc[idx, "patient_id"]
        image_id = self.data.loc[idx, "image_id"]
        image_path = os.path.join(
            self.images_path, str(patient_id), str(image_id) + ".dcm"
        )

        img = dcmread(image_path).pixel_array
        img = self.transform(img)

        if self.additional is None:
            return img
        else:
            return img, *[self.df[idx][i] for i in self.additional]
        