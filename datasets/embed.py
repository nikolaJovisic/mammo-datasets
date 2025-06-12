from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset
from enum import Enum, auto
from utils.image_tiles import tile_multiple

class EmbedReturnMode(Enum):
    IMAGE_ONLY = auto() # (image)
    IMAGE_ASSES = auto() # (image, asses), drops rows with no asses
    PAIR_ASSES = auto() # (CC, MLO, asses), drops rows with no asses, CC/MLO are None if not present
    TILES_ASSES = auto() # (CC, MLO, list of tiles of inside of breast, asses), drops rows with no asses, CC/MLO are None if not present

def _aggregate_asses_by_breast(df, asses):
    df = df.dropna(subset=['asses'])
    df = df[df['ViewPosition'].isin(['CC', 'MLO'])]

    paths = df.pivot_table(
        index=['acc_anon', 'ImageLateralityFinal'],
        columns='ViewPosition',
        values='png_path',
        aggfunc='first'
    ).reset_index().rename(columns={'CC': 'png_path_CC', 'MLO': 'png_path_MLO'})

    asses = df.groupby(['acc_anon', 'ImageLateralityFinal'])['asses'].max().reset_index()
    df = pd.merge(paths, asses, on=['acc_anon', 'ImageLateralityFinal'])


    for col in ['png_path_CC', 'png_path_MLO']:
        df[col] = df[col].where(df[col].notna(), None).astype(object)

    return df

def _get_df(csv_path, return_mode, asses):
    df = pd.read_csv(csv_path)
    
    if asses is not None and not isinstance(asses, (list, set, tuple)):
        asses = [asses]
    
    if return_mode == EmbedReturnMode.PAIR_ASSES or return_mode == EmbedReturnMode.TILES_ASSES:
        df = _aggregate_asses_by_breast(df, asses)
    
    if asses is not None:
        df = df[df['asses'].isin(asses)]
    
    return df

class EmbedDataset(Dataset):
    def __init__(self, csv_path: str, images_path: str, transform=lambda x: x, return_mode=EmbedReturnMode.IMAGE_ONLY, 
                 asses=None, tile_size=None, tile_overlap=None):
        self.df = _get_df(csv_path, return_mode, asses)
        self.images_path = images_path
        self.transform = transform
        self.return_mode = return_mode
        self.asses = asses
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        if return_mode == EmbedReturnMode.TILES_ASSES and (tile_size is None or tile_overlap is None):
            raise ValueError("tile_size and tile_overlap must be provided for TILES_ASSES mode")

        print(f'Embed dataset initialized with {len(self.df)} rows.')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self._get_ret_val(row)

    def _get_ret_val(self, row):
        if self.return_mode in [EmbedReturnMode.IMAGE_ONLY, EmbedReturnMode.IMAGE_ASSES]:
            img_path = os.path.join(self.images_path, row['png_path'])
            img = Image.open(img_path)
            img = self.transform(img)
            if self.return_mode == EmbedReturnMode.IMAGE_ONLY:
                return img
            return img, row['asses']

        cc = mlo = None
        if row['png_path_CC'] is not None:
            cc_path = os.path.join(self.images_path, row['png_path_CC'])
            cc = Image.open(cc_path)
            cc = self.transform(cc)

        if row['png_path_MLO'] is not None:
            mlo_path = os.path.join(self.images_path, row['png_path_MLO'])
            mlo = Image.open(mlo_path)
            mlo = self.transform(mlo)

        if self.return_mode == EmbedReturnMode.PAIR_ASSES:
            return cc, mlo, row['asses']

        if self.return_mode == EmbedReturnMode.TILES_ASSES:
            images = [img for img in (cc, mlo) if img is not None]
            tiles = tile_multiple(images, self.tile_size)
            return cc, mlo, tiles, row['asses']

