import numpy as np
from utils.preprocess import get_breast_mask, prepare_for_preprocess, negate_if_should, flip_if_should

def tile_single(image, tile_size=(518, 518), overlap=0.25, threshold=0.05, tile_increase_tolerance=0.1):
    image = prepare_for_preprocess(image)
    image = negate_if_should(image)
    image = flip_if_should(image)
    mask = get_breast_mask(image)
    
    h, w = image.shape
    base_tile_h, base_tile_w = tile_size

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    region_h = y_max - y_min + 1
    region_w = x_max - x_min + 1

    best_tile_h = base_tile_h
    best_tile_w = base_tile_w
    
    for scale_h in np.linspace(1.0, 1.0 + tile_increase_tolerance, 10):
        tile_h = int(base_tile_h * scale_h)
        num_tiles_h = max(1, int(np.ceil((region_h - tile_h * overlap) / (tile_h * (1 - overlap)))))
        stride_h = (region_h - tile_h) / max(1, num_tiles_h - 1) if num_tiles_h > 1 else 0
        stride_h = int(np.floor(stride_h))
        
        if stride_h >= 0:
            best_tile_h = tile_h
    
    for scale_w in np.linspace(1.0, 1.0 + tile_increase_tolerance, 10):
        tile_w = int(base_tile_w * scale_w)
        num_tiles_w = max(1, int(np.ceil((region_w - tile_w * overlap) / (tile_w * (1 - overlap)))))
        stride_w = (region_w - tile_w) / max(1, num_tiles_w - 1) if num_tiles_w > 1 else 0
        stride_w = int(np.floor(stride_w))
        
        if stride_w >= 0:
            best_tile_w = tile_w

    tile_h = best_tile_h
    tile_w = best_tile_w

    num_tiles_h = max(1, int(np.ceil((region_h - tile_h * overlap) / (tile_h * (1 - overlap)))))
    num_tiles_w = max(1, int(np.ceil((region_w - tile_w * overlap) / (tile_w * (1 - overlap)))))

    stride_h = (region_h - tile_h) / max(1, num_tiles_h - 1) if num_tiles_h > 1 else 0
    stride_w = (region_w - tile_w) / max(1, num_tiles_w - 1) if num_tiles_w > 1 else 0
    
    stride_h = int(np.floor(stride_h))
    stride_w = int(np.floor(stride_w))
    
    y_start = max(0, y_min)
    x_start = max(0, x_min)

    tiles = []
    for i in range(num_tiles_h):
        y = y_start + i * stride_h
        if y + tile_h > h:
            y = h - tile_h
        for j in range(num_tiles_w):
            x = x_start + j * stride_w
            if x + tile_w > w:
                x = w - tile_w
            tile_img = image[y:y + tile_h, x:x + tile_w]
            if tile_img.mean() > threshold * image.max():
                tiles.append(tile_img)

    return tiles








def tile_multiple(images, tile_size=(518, 518), overlap=0.25):
    tiles = []
    for image in images:
        img_tiles = tile_single(image, tile_size, overlap)
        tiles.extend(img_tiles)
    return tiles

