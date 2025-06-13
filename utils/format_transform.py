from enum import Enum, auto
import torch
import numpy as np

def get_imagenet_normalization():
    return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class ConvertTo(Enum):
    UINT8 = auto()
    FLOAT32 = auto()
    TENSOR = auto()
    RGB_TENSOR = auto()
    RGB_TENSOR_NORM = auto()
    RGB_TENSOR_IMGNET_NORM = auto()
    
_CONVERT_FROM_UINT8_TO = {
    ConvertTo.UINT8: lambda x, _: x,
    ConvertTo.FLOAT32: lambda x, _: (x / 255).astype(np.float32),
    ConvertTo.TENSOR: lambda x, _: torch.from_numpy(x / 255).float(),
    ConvertTo.RGB_TENSOR: lambda x, _: torch.from_numpy(np.stack([x] * 3) / 255).float(),
    ConvertTo.RGB_TENSOR_NORM: lambda x, norm: _rgb_tensor_normalized(x, norm),
    ConvertTo.RGB_TENSOR_IMGNET_NORM: lambda x, _: _rgb_tensor_normalized(x, get_imagenet_normalization())
}

def _rgb_tensor_normalized(x, normalization):
    img = torch.from_numpy(np.stack([x]*3) / 255).float()
    mean = torch.tensor(normalization[0]).view(3, 1, 1)
    std = torch.tensor(normalization[1]).view(3, 1, 1)
    return (img - mean) / std

    
class FormatTransform:
    def __init__(
        self,
        convert_to=ConvertTo.UINT8,
        normalization=None
    ):
        self.convert_to = convert_to
        
        if convert_to == ConvertTo.RGB_TENSOR_NORM and normalization is None:
            raise ValueError(f"Set normalization parameter for this type of conversion.")
        
        self.normalization = normalization

    def __call__(self, img):
        return _CONVERT_FROM_UINT8_TO[self.convert_to](img, self.normalization)

    