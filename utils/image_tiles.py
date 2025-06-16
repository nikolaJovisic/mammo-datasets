import numpy as np
from utils.preprocess import get_breast_mask, prepare_for_preprocess, negate_if_should, flip_if_should

def tile_single(image, tile_size=(518, 518), overlap=0.25, threshold=0.05):
    image = prepare_for_preprocess(image)
    image = negate_if_should(image)
    image = flip_if_should(image)
    mask = get_breast_mask(image)
    
    h, w = image.shape
    tile_h, tile_w = tile_size
    stride_h = int(tile_h * (1 - overlap))
    stride_w = int(tile_w * (1 - overlap))

    ys, xs = np.where(mask)
    
    if len(ys) == 0:
        return []
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    y_start = max(0, (y_min // stride_h) * stride_h)
    y_end   = min(h, ((y_max // stride_h) + 1) * stride_h + tile_h)
    x_start = max(0, (x_min // stride_w) * stride_w)
    x_end   = min(w, ((x_max // stride_w) + 1) * stride_w + tile_w)

    tiles = []
    for y in range(y_start, y_end - tile_h + 1, stride_h):
        for x in range(x_start, x_end - tile_w + 1, stride_w):
            tile_img = image[y:y + tile_h, x:x + tile_w]
            
            if tile_img.mean() > threshold * image.max():
                tiles.append(tile_img)

    return tiles


def tile_multiple(images, tile_size=(518, 518), overlap=0.25):
    tiles = []
    for image in images:
        img_tiles = tile_single(image, mask, tile_size, overlap)
        tiles.extend(img_tiles)
    return tiles

