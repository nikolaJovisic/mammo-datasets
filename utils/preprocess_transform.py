import numpy as np
import torch
from enum import Enum, auto
from utils.preprocess import (
    keep_only_breast,
    get_breast_mask,
    otsu_cut,
    pad,
    apply_clahe,
    flip_if_should,
    negate_if_should,
    resize_img,
    resize_mask
)
from windowing.apply import window as apply_windowing

def to_ndarray(img):
    return img if isinstance(img, np.ndarray) else np.asarray(img)

class PreprocessTransform:
    def __init__(
        self,
        dicom=True,
        clahe=False,
        return_mask=False,
        aspect_ratio=1 // 1,
        resize=None
    ):
        self.dicom = dicom
        self.clahe = clahe
        self.return_mask = return_mask
        self.aspect_ratio = aspect_ratio
        self.resize = (resize, resize) if isinstance(resize, int) else resize

    def __call__(self, img, window=(None, None)):
        if self.dicom:
            img = img.pixel_array
        
        img = to_ndarray(img)
        img = negate_if_should(img)
        
        if None in window:
            window = img.min(), img.max()
        
        img = apply_windowing(img, *window)
        
        img = flip_if_should(img)
        img, mask = keep_only_breast(img)

        if self.clahe:
            img = apply_clahe(img)
            img = img * mask

        img = otsu_cut(img)
        img = pad(img, self.aspect_ratio)

        if self.resize is not None:
            img = resize_img(img, self.resize)

        return (img, get_breast_mask(img)) if self.return_mask else img
