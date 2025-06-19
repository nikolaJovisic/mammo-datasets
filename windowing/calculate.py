from utils.preprocess import prepare_for_preprocess, negate_if_should, flip_if_should, binarize, resize_img, fill_holes
from skimage.transform import resize
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from math import pi, sqrt
from windowing.apply import window
from tqdm import tqdm
import matplotlib.pyplot as plt

def _get_dicom_mask(image):
    image = prepare_for_preprocess(image)
    mask = binarize(image)
    mask = fill_holes(mask)
    return mask


def _largest_square_in_mask(image: np.ndarray, mask: np.ndarray, downscale: int = 16) -> np.ndarray:
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions do not match.")

    h, w = mask.shape
    if downscale > 1:
        small_mask = resize(mask, (h // downscale, w // downscale), order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)
    else:
        small_mask = mask

    small_mask = small_mask != 0
    sh, sw = small_mask.shape
    dp = np.zeros((sh, sw), dtype=np.int32)

    max_size = 0
    max_pos = (0, 0)

    for i in range(sh):
        for j in range(sw):
            if small_mask[i, j]:
                if i == 0 or j == 0:
                    dp[i, j] = 1
                else:
                    dp[i, j] = min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]) + 1
                if dp[i, j] > max_size:
                    max_size = dp[i, j]
                    max_pos = (i, j)

    if max_size == 0:
        return None

    i, j = max_pos
    y1 = (i - max_size + 1) * downscale
    x1 = (j - max_size + 1) * downscale
    size = max_size * downscale

    y2 = min(y1 + size, h)
    x2 = min(x1 + size, w)

    y1 = max(0, y2 - size)
    x1 = max(0, x2 - size)

    return image[y1:y2, x1:x2]

def _apply_gabor_filter(image, frequency, theta, sigma):
    image = image.astype(np.float32)
    image = (image - image.mean()) / (image.std() + 1e-8)
    kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
    real = ndi.convolve(image, np.real(kernel), mode='wrap')
    imag = ndi.convolve(image, np.imag(kernel), mode='wrap')
    return np.sqrt(real**2 + imag**2)

def _get_all_filtered(image):
    all_filtered = []
    for frequency in [1/8, 1/4]: # simplified
#     for frequency in [1/8, sqrt(2)/8, 1/4]: # for original paper implementation
        sigma = 1/(2*frequency)
        for theta in [0, pi/6, 2*pi/3]: # simplified 
#         for theta in [0, pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6]: # for original paper implementation
            filtered = _apply_gabor_filter(image, frequency, theta, sigma)
            all_filtered.append(filtered)
    return all_filtered

def _mutual_information(image_12bit, image_8bit, bins_12=4096, bins_8=256):
    joint_hist, _, _ = np.histogram2d(
        image_12bit.ravel(),
        image_8bit.ravel(),
        bins=(bins_12, bins_8)
    )
    joint_prob = joint_hist / joint_hist.sum()

    p_I = np.sum(joint_prob, axis=1)
    p_I_tilde = np.sum(joint_prob, axis=0)

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    H_I = entropy(p_I)
    H_I_tilde = entropy(p_I_tilde)
    H_joint = entropy(joint_prob)

    mi = H_I + H_I_tilde - H_joint
    return mi, joint_hist, p_I, p_I_tilde


def _evaluate(orig, orig_filtered, a, b):
    windowed = window(orig, a, b)
    windowed_filtered = _get_all_filtered(windowed)
    mis = []
    for single_orig_filtered, single_windowed_filtered in zip(orig_filtered, windowed_filtered):
        mi, joint_hist, p_I, p_I_tilde = _mutual_information(single_orig_filtered, single_windowed_filtered, bins_12 = 4096 if orig.max() < 4096 else 65536)
        mis.append(mi)
    return np.average(mis)

def _center_crop(img, crop_size) -> np.ndarray:
    h, w = img.shape
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    return img[start_y:start_y + crop_size, start_x:start_x + crop_size]

def optimize(img, steps=30):
    all_filtered_img = _get_all_filtered(img)
    
    min_val = img.min()
    max_val = img.max()
    
    best_a = min_val
    best_b = max_val
    best_eval = 0
    
    perc10 = np.uint16(np.percentile(img, 10))
    perc90 = np.uint16(np.percentile(img, 90))
    
    step_a = (perc10 - min_val) // steps
    if step_a == 0:
        best_a = perc10
    else:
        for a in range(min_val, perc10, step_a):
            val = _evaluate(img, all_filtered_img, a, best_b)
            if val > best_eval:
                best_a = a
                best_eval = val
    
    step_b = (max_val - perc90) // steps
    if step_b == 0:
        best_b = perc90
    else:
        for b in range(perc90, max_val, step_b):
            val = _evaluate(img, all_filtered_img, best_a, b)
            if val > best_eval:
                best_b = b
                best_eval = val
    
    return best_a, best_b



def crop_center_and_corners(img, mid_size, crop_shape):
    top_left = resize_img(img[:mid_size, :mid_size], crop_shape)
    top_right = resize_img(img[:-mid_size, :mid_size], crop_shape)
    
    center = resize_img(_center_crop(img, mid_size), crop_shape)
    
    bottom_left = resize_img(img[mid_size:, -mid_size:], crop_shape)
    bottom_right = resize_img(img[-mid_size:, -mid_size:], crop_shape)
    
    return [top_left, top_right, center, bottom_left, bottom_right]

def take_crops(img, crop_size=128):
    crop_shape = 2*(crop_size,)                      
    
    crops = [resize_img(img, crop_shape)]
    
    if img.shape[0] > 4 * crop_size:
        mid_size = int((img.shape[0] - crop_size) / 2)
        crops.extend(crop_center_and_corners(img, mid_size, crop_shape))
        
    crops.extend(crop_center_and_corners(img, crop_size, crop_shape))
    
    return crops
    
    
    
def calculate_a_b(pixel_array, steps=10):
    img = negate_if_should(pixel_array)
    mask = _get_dicom_mask(img)
    img = _largest_square_in_mask(img, mask)
    
    if img.shape[0] < 512: # not enough tissue has been detected to perform optimization
        return np.nan, np.nan
    
    crops = take_crops(img)
    
#     n = len(crops)
#     plt.figure(figsize=(n * 3, 3))
#     for i, crop in enumerate(crops):
#         print(crop.shape)
#         plt.subplot(1, n, i + 1)
#         plt.imshow(crop, cmap='gray')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    
    a = np.inf
    b = 0
    
    for crop in crops:
        a_cand, b_cand = optimize(crop)
        if a_cand < a:
            a = a_cand
        if b_cand > b:
            b = b_cand
    return a, b

def calculate_a_b_3ch(pixel_array):
    max_val = np.array(pixel_array).max()
    a, b = calulate_a_b(pixel_array)
    a_narrowed = a + np.uint16(0.3 * a)
    b_narrowed = b - (0.3 * np.uint16(max_val - b))
    a_widen = a - np.uint16(0.3 * a)
    b_widen = b + (0.3 * np.uint16(max_val - b))
    return np.array([[a_narrowed, b_narrowed], [a, b], [a_widen, b_widen]])