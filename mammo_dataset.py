import os
from pathlib import Path
from enum import Enum, auto

from PIL import Image
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from utils.image_tiles import tile_single, tile_multiple
from utils.aggregations import aggregate_by_breast
from utils.preprocess import resize_img
from utils.preprocess_transform import PreprocessTransform
from utils.format_transform import FormatTransform, ConvertTo
from utils.to_dict import dataset_to_dict

from dataset_specifics import FileFormat

from datasets.embed import EMBEDSpecifics
from datasets.rsna import RSNASpecifics
from datasets.vindr import VINDRSpecifics
from datasets.csaw import CSAWSpecifics


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
    IMAGE_ONLY = "image_only" # (image)
    IMAGE_MASK = "image_mask" # (image, breast_mask)
    IMAGE_LABEL = "image_label" # (image, label), drops rows with no labels
    IMAGE_RAW = "image_raw" # (raw), raw dicom/PIL.Image object
    IMAGE_RAW_NUMPY = "image_raw_numpy" # (ndarray), raw image data, no preprocessing and formatting
    BREAST_LABEL = "breast_label" # (list of images of breast for a study, label), drops rows with no labels
    BREAST_TILES_LABEL = "breast_tiles_label" # (list of images of breast for a study, list of tiles of inside of breast of all images, label), drops rows with no labels


class MammoDataset(Dataset):
    def __init__(self, dataset=DatasetEnum.EMBED, split=Split.ALL, return_mode=ReturnMode.IMAGE_ONLY,
                 convert_to=ConvertTo.UINT8, labels=None, clahe=False, aspect_ratio=1//1, resize=None, 
                 tile_size=None, tile_overlap=None, final_resize=None, return_index=False, limit=None, 
                 read_window=False, cfg_path=None):
        
        if 'tiles' in return_mode.value:
            if tile_size is None:
                raise ValueError(f"Set tile_size for return mode {return_mode}.")
            if tile_overlap is None:
                tile_overlap = 0.25
                
        if isinstance(final_resize, int):
            final_resize = (final_resize,)*2
                
        self.ds_spec = _DATASETS_MAP[dataset]()
        
        if cfg_path is None:
            cfg_path = Path(__file__).parent / "config.yaml"
        
        self.cfg_path = cfg_path

        self.cfg = OmegaConf.load(self.cfg_path)
        
        self.split = split
        self.csv_path = self.cfg[dataset.value][f'{split.value}csv_path']
        self.images_path = self.cfg[dataset.value]['images_path']
        
        is_dicom = self.ds_spec.file_format == FileFormat.DICOM
        
        self.preprocess_transform = PreprocessTransform(is_dicom, clahe, return_mode==ReturnMode.IMAGE_MASK,
                                                        aspect_ratio, resize)
        
        normalization = None
        
        if convert_to == ConvertTo.RGB_TENSOR_NORM:
            normalization = self.ds_spec.normalization_stats
        
        self.format_transform = FormatTransform(convert_to, normalization, final_resize)
        
        self.return_mode = return_mode
        
        if labels is not None and not isinstance(labels, (list, set, tuple)):
            labels = [labels]
                    
        self.labels = labels
        
        self.limit = limit
        
        self.df = self._get_df()
        
        if isinstance(tile_size, int):
            tile_size = (tile_size,)*2
            
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
        self.read_window = read_window
        self.return_index = return_index
        
        
        print(f'Dataset {dataset.value} initialized with {len(self.df)} rows.')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, pd.Series)):
            return [self._get_ret_val(self.df.iloc[i]) for i in idx]
        if isinstance(idx, slice):
            return [self._get_ret_val(row) for _, row in self.df.iloc[idx].iterrows()]
        return self._get_ret_val(self.df.iloc[idx])
    
    def _load_preprocess_from_path(self, path, window):
        img_path = os.path.join(self.images_path, path)
        img = self.ds_spec.read_file(img_path)
        return self.preprocess_transform(img, window)
        
    def _get_ret_val(self, row):
        output = self._prepare_output(row)
        if self.return_index:
            return row["csv_index"], output
        return output
        
    def _prepare_output(self, row):
        return self._image_row_output(row) if self.return_mode.value.startswith('image') else self._breast_row_output(row)
        
    def _image_row_output(self, row):
        img_path = os.path.join(self.images_path, row[self.ds_spec.path_col])
        img = self.ds_spec.read_file(img_path)
        window = (row['dcm_window_a'], row['dcm_window_b']) if self.read_window else (None, None)

        if self.return_mode == ReturnMode.IMAGE_RAW:
            return img

        if self.return_mode == ReturnMode.IMAGE_RAW_NUMPY:
            return self.ds_spec.raw_numpy(img)

        preprocessed = self.preprocess_transform(img, window)

        if self.return_mode == ReturnMode.IMAGE_MASK:
            img, mask = preprocessed
            return self.format_transform(img), mask

        if self.return_mode == ReturnMode.IMAGE_LABEL:
            return self.format_transform(preprocessed), self.ds_spec.map_label(row[self.ds_spec.label_col])

        return self.format_transform(preprocessed)

    def _breast_row_output(self, row):
        images = []
        for path, window in zip(row[self.ds_spec.path_col], row['window']):
            if not self.read_window:
                window = (None, None)
            images.append(self._load_preprocess_from_path(path, window))
            
        if self.return_mode == ReturnMode.BREAST_LABEL:
            images[:] = map(self.format_transform, images)
            return images, self.ds_spec.map_label(row[self.ds_spec.label_col])
        
        if self.return_mode == ReturnMode.BREAST_TILES_LABEL: 
            tiles = tile_multiple(images, self.tile_size, self.tile_overlap)
            images[:] = map(self.format_transform, images)
            tiles[:] = map(self.format_transform, tiles)
            return images, tiles, self.ds_spec.map_label(row[self.ds_spec.label_col])
        
        raise NotImplementedError
    
    def _get_df(self):
        df = pd.read_csv(self.csv_path)
        df.reset_index(drop=False, inplace=True)
        df.rename(columns={"index": "csv_index"}, inplace=True)
        
        if self.return_mode.value.startswith('breast'):
            df = aggregate_by_breast(df, *self.ds_spec.get_agg_columns())

        if self.labels is not None:
            df = df[df[self.ds_spec.label_col].isin(self.labels)]
            
        if self.limit is not None:
            df = df.iloc[:self.limit].reset_index(drop=True)
        
        return df
    
    def to_dict(self):
        return dataset_to_dict(self)