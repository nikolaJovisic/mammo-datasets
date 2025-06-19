import numpy as np

def window(image, lower_bound, upper_bound):
    image = np.asarray(image)
    image = np.clip(image, lower_bound, upper_bound)
    image = ((image - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
    return image