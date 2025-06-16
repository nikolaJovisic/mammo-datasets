import cv2
import numpy as np
from skimage.filters import threshold_otsu

def prepare_for_preprocess(img):
    if hasattr(img, 'dtype') or img.dtype == np.uint8:
        return img
    
    ret_img = np.array(image, dtype=np.float32)
    ret_img = 255 * (ret_img - np.min(ret_img)) / (np.max(ret_img) - np.min(ret_img))
    return ret_img.astype(np.uint8)
        

def add_one_if_even(number):
    if number % 2 == 0:
        return number + 1
    else:
        return number

def create_kernel(img, factor):
    return add_one_if_even(img.shape[0] // factor), add_one_if_even(img.shape[1] // factor)

def binarize(img):
    b_img = img.astype(np.float32)
    b_img = 255 * (b_img - np.min(b_img)) / (np.max(b_img) - np.min(b_img))
    b_img = b_img.astype(np.uint8)

    blured = cv2.GaussianBlur(b_img, create_kernel(b_img, 100), 0)

    otsu_tr = threshold_otsu(blured) * 0.5 #0.175
    mask = np.where(blured >= otsu_tr, 1, 0).astype(np.uint8)

    return mask


def dilate(mask, factor):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, create_kernel(mask, factor))
    dilated_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return dilated_mask

def erode(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, create_kernel(mask, 500))
    eroded_mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

    return eroded_mask

def keep_largest_blob(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_blob_mask = np.zeros_like(mask)
    cv2.drawContours(largest_blob_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

    return largest_blob_mask


def get_breast_mask(image, dilation_factor=10):
    mask = binarize(image)
    mask = erode(mask)
    mask = dilate(mask, dilation_factor)
    return keep_largest_blob(mask)


def keep_only_breast(image):
    mask = get_breast_mask(image, 10)
    masked_image = image * mask
    return masked_image, mask

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1)
    return clahe.apply(image)


def crop_borders(image):
    l = 0.01
    r = 0.01
    u = 0.04
    d = 0.04

    n_rows, n_cols = image.shape

    l_crop = int(n_cols * l)
    r_crop = int(n_cols * (1 - r))
    u_crop = int(n_rows * u)
    d_crop = int(n_rows * (1 - d))

    cropped_img = image[u_crop:d_crop, l_crop:r_crop]

    return cropped_img, (l_crop, n_cols - r_crop, u_crop, n_rows - d_crop)


def should_flip(image):
    x_center = image.shape[1] // 2
    col_sum = image.sum(axis=0)

    left_sum = np.sum(col_sum[0:x_center])
    right_sum = np.sum(col_sum[x_center:-1])

    return left_sum < right_sum

def flip_if_should(image):
    return np.fliplr(image) if should_flip(image) else image

def pad(image, ar=1152/896):
    n_rows, n_cols = image.shape
    image_ratio = n_rows / n_cols
    if image_ratio == ar:
        return image
    if ar < image_ratio:
        new_n_cols = int(n_rows / ar)
        ret_val = np.zeros(shape=(n_rows, new_n_cols)).astype(image.dtype)
    else:
        new_n_rows = int(n_cols * ar)
        ret_val = np.zeros(shape=(new_n_rows, n_cols)).astype(image.dtype)
    ret_val[:n_rows, :n_cols] = image
    return ret_val

def negate_if_should(image):
    num_bins = 20
    hist, bins = np.histogram(image.ravel(), bins=num_bins, range=[image.min(), image.max()])
    max_bin = np.argmax(hist)

    return image if max_bin < (num_bins / 2) else np.max(image) - image



def preprocess_scan_with_mask(image, mass_mask):
    image, borders = crop_borders(image)
    image = negate_if_should(image)
    image, breast_mask = keep_only_breast(image)
    flip = should_flip(image)

    if flip:
        image = np.fliplr(image)
        breast_mask = np.fliplr(breast_mask)
        mass_mask = np.fliplr(mass_mask)

    shape_before_padding = image.shape
    image = pad(image)
    mass_mask = pad(mass_mask)

    spatial_changes = (borders, flip, shape_before_padding, image.shape)

    return image, mass_mask, spatial_changes


def preprocess_scan(image):
    image, _, spatial_changes = preprocess_scan_with_mask(image, image)
    return image, spatial_changes

def get_edge_contours(image):
    mask = get_breast_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_image = np.zeros_like(mask)
    cv2.drawContours(edge_image, contours, -1, (1), 1)
    return edge_image

def reverse_spatial_changes(image, spatial_changes):
    borders, flip, shape_before_padding, shape = spatial_changes
    l, r, u, d = borders
    image = cv2.resize(src=image, dsize=shape, interpolation=cv2.INTER_NEAREST)
    image = image[: shape_before_padding[0], : shape_before_padding[1]]
    if flip:
        image = np.fliplr(image)
    image = np.pad(image, ((u, d), (l, r)))
    return image

def otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def otsu_cut(img):
    mask = otsu_mask(img)
        
    if mask.size == 0 or not np.any(mask):
        return Image.fromarray(x)

    rows = np.any(mask == 255, axis=1)
    cols = np.any(mask == 255, axis=0)

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    x = img[min_row : max_row + 1, min_col : max_col + 1]
    return x

def resize_img(img, resize):
    return cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)

def resize_mask(mask, resize):
    return cv2.resize(mask, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)