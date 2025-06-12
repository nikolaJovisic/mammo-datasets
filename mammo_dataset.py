from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset
from enum import Enum, auto
from utils.image_tiles import tile_multiple
from utils.aggregations import aggregate_by_breast
from utils.preprocess_transform import PreprocessTransform

from datasets.embed import EMBEDSpecifics
from datasets.rsna import RSNASpecifics
from datasets.vindr import VINDRSpecifics
from datasets.csaw import CSAWSpecifics

from omegaconf import OmegaConf

class DatasetEnum(Enum):
    EMBED = "embed"
    RSNA = "rsna"
    VINDR = "vindr"
    CSAW = "csaw"
    
_DATASETS_MAP = {
    DatasetEnum.EMBED: EMBEDSpecifics,
    DatasetEnum.RSNA: RSNASpecifics,
    DatasetEnum.VINDR: VINDRSpecifics,
    DatasetEnum.CSAW: CSAWSpecifics
}
    
class Split(Enum):
    TRAIN = "train_"
    VALID = "valid_"
    TEST = "test_"
    ALL = ""
    

class ReturnMode(Enum):
    IMAGE_ONLY = auto() # (image)
    IMAGE_LABEL = auto() # (image, LABEL), drops rows with no labels
    CC_MLO_LABEL = auto() # (CC, MLO, LABEL), drops rows with no labels, CC/MLO are None if not present
    CC_MLO_TILES_LABEL = auto() # (CC, MLO, list of tiles of inside of breast, label), drops rows with no labels, CC/MLO are None if not present    


class MammoDataset(Dataset):
    def __init__(self, dataset=DatasetEnum.EMBED, split=Split.TRAIN, return_mode=ReturnMode.IMAGE_ONLY, 
                 labels=None, tile_size=(1024, 1024), tile_overlap=0.25, cfg_path='config.yaml'):
        self.ds_spec = _DATASETS_MAP[dataset]()
        
        cfg = OmegaConf.load(cfg_path)
        
        self.csv_path = cfg[dataset.value][f'{split.value}csv_path']
        self.images_path = cfg[dataset.value]['images_path']
        
        
        self.transform = PreprocessTransform()
        self.return_mode = return_mode
        
        
        if labels is not None and not isinstance(labels, (list, set, tuple)):
            labels = [labels]
            
        self.labels = labels
        
        self.df = self._get_df()
        
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        print(f'Dataset {dataset.value} initialized with {len(self.df)} rows.')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._get_ret_val(row) for _, row in self.df.iloc[idx].iterrows()]
        elif isinstance(idx, (list, tuple, pd.Series)):
            return [self._get_ret_val(self.df.iloc[i]) for i in idx]
        else:
            row = self.df.iloc[idx]
            return self._get_ret_val(row)


    def _get_ret_val(self, row):
        if self.return_mode in [ReturnMode.IMAGE_ONLY, ReturnMode.IMAGE_LABEL]:
            
            img_path = os.path.join(self.images_path, row[self.ds_spec.path_col])
            img = self.ds_spec.load_img(img_path)
            img = self.transform(img)
            if self.return_mode == ReturnMode.IMAGE_ONLY:
                return img
            return img, row[self.ds_spec.label_col]

        cc = mlo = None
        if row[self.ds_spec.cc_col] is not None:
            cc_path = os.path.join(self.images_path, row[self.ds_spec.cc_col])
            cc = self.ds_spec.load_img(cc_path)
            cc = self.transform(cc)

        if row[self.ds_spec.mlo_col] is not None:
            mlo_path = os.path.join(self.images_path, row[self.ds_spec.mlo_col])
            mlo = self.ds_spec.load_img(mlo_path)
            mlo = self.transform(mlo)

        if self.return_mode == ReturnMode.CC_MLO_LABEL:
            return cc, mlo, row[self.ds_spec.label_col]

        if self.return_mode == ReturnMode.CC_MLO_TILES_LABEL:
            images = [img for img in (cc, mlo) if img is not None]
            tiles = tile_multiple(images, self.tile_size)
            return cc, mlo, tiles, row[self.ds_spec.label_col]
    
    def _get_df(self):
        df = pd.read_csv(self.csv_path)

        if self.return_mode == ReturnMode.CC_MLO_LABEL or self.return_mode == ReturnMode.CC_MLO_TILES_LABEL:
            df = aggregate_by_breast(df, *self.ds_spec.get_agg_columns())

        if self.labels is not None:
            df = df[df[self.ds_spec.label_col].isin(self.labels)]

        return df

