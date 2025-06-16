import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess_dicom(dcm):
    img = dcm.pixel_array #.astype(np.float32)
#     img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    
#     plt.hist(img)
#     plt.show()

#     percentiles = [1, 5, 10, 50, 90, 95, 99, 100]
#     values = np.percentile(img.astype(np.float32), percentiles)

#     plt.hist(img.ravel(), bins=16, color='gray', alpha=0.7)
#     for p, v in zip(percentiles, values):
#         plt.axvline(x=v, linestyle='--', label=f'{p}th: {v:.1f}')
#     plt.legend()
#     plt.title("Histogram with Percentile Markers")
#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")
#     plt.show()

    
#     img *= getattr(dcm, "RescaleSlope", 1.0)
#     img += getattr(dcm, "RescaleIntercept", 0.0)
    
#     if getattr(dcm, "PhotometricInterpretation", None) == "MONOCHROME1":
#         img = np.max(img) - img
    
#     if hasattr(dcm, "WindowCenter") and hasattr(dcm, "WindowWidth"):
#         wc = dcm.WindowCenter
#         ww = dcm.WindowWidth
        
#         if isinstance(wc, (list, tuple)) or hasattr(wc, '__len__'):
#             wc = wc[0]
#         if isinstance(ww, (list, tuple)) or hasattr(ww, '__len__'):
#             ww = ww[0]

#         wc = float(wc)
#         ww = float(ww)
        
#         min_val = wc - ww / 2
#         max_val = wc + ww / 2
        
#         img = np.clip(img, min_val, max_val)
#         img = (img - min_val) / (max_val - min_val) * 255
    
    return img #.astype(np.uint8)

