import numpy as np
import torch
from enum import Enum, auto
from utils.preprocess import (
    keep_only_breast,
    otsu_cut,
    pad,
    apply_clahe,
    flip_if_should,
    negate_if_should,
    resize_img,
    resize_mask
)

def to_ndarray(img):
    return img if isinstance(img, np.ndarray) else np.asarray(img)


class ConvertFrom(Enum):
    UINT8 = auto()
    UINT16 = auto()
    MINMAX = auto()


_CONVERT_TO_UINT8_FROM = {
    ConvertFrom.UINT8: lambda x: x,
    ConvertFrom.UINT16: lambda x: (x // 255).astype(np.uint8),
    ConvertFrom.MINMAX: lambda x: (255 * ((x - x.min()) / (x.max() - x.min()))).astype(np.uint8)
}


class PreprocessTransform:
    def __init__(
        self,
        clahe=False,
        return_mask=False,
        aspect_ratio=1 // 1,
        resize=None,
        convert_from=ConvertFrom.MINMAX,
    ):
        self.clahe = clahe
        self.return_mask = return_mask
        self.aspect_ratio = aspect_ratio
        self.resize = resize
        self.convert_from = convert_from

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

        if self.resize is not None:
            img = resize_img(img, self.resize)
            mask = resize_mask(mask, self.resize)

        return (img, mask) if self.return_mask else img
