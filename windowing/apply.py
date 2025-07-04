import numpy as np

def window(image, lower_bound, upper_bound):
    image = np.asarray(image)
    img_min = image.min()
    img_max = image.max()
    lower_bound = max(lower_bound, img_min)
    upper_bound = min(upper_bound, img_max)
    image = np.clip(image, lower_bound, upper_bound)
    image = ((image - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
    return image
