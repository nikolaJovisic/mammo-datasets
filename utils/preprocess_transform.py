import numpy as np
import torch
from utils.preprocess import keep_only_breast, otsu_cut, pad, apply_clahe, flip_if_should, negate_if_should

def to_ndarray(img):
    return img if isinstance(img, np.ndarray) else np.asarray(img)

_CONVERT_TO_UINT8_FROM = {
    "uint8": lambda x: x,
    "uint16" : lambda x: (x // 255).astype(np.uint8),
    "minmax" : lambda x: (255 * ((x - x.min()) / (x.max() - x.min()))).astype(np.uint8)
}

_CONVERT_FROM_UINT8_TO = {
    "uint8": lambda x: x,
    "float32": lambda x: (x / 255).astype(np.float32),
    "tensor": lambda x: torch.from_numpy(x / 255)
}

class PreprocessTransform:
    def __init__(self, clahe=False, return_mask=False, aspect_ratio=1//1, convert_from="minmax", convert_to="uint8"):
        self.clahe = clahe
        self.return_mask = return_mask
        self.aspect_ratio = aspect_ratio
        self.convert_from = convert_from
        self.convert_to = convert_to

    def __call__(self, img):
        img = to_ndarray(img)
        img = _CONVERT_TO_UINT8_FROM[self.convert_from](img)
        
        img = negate_if_should(img)
        img = flip_if_should(img)
        img, mask = keep_only_breast(img)
        
        if self.clahe:
            img = apply_clahe(img)
            img = img * mask
            
        img = otsu_cut(img)
        img = pad(img, self.aspect_ratio)
        
        img = _CONVERT_FROM_UINT8_TO[self.convert_to](img)
        
        return (img, mask) if self.return_mask else img