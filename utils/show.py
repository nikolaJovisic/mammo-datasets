import matplotlib.pyplot as plt

def siar(images, limit=None):
    import matplotlib.pyplot as plt
    import numpy as np
    
    if isinstance(images, np.ndarray):
        images = [images]
    
    n = len(images)
    if limit is not None:
        n = min(n, limit)
        images = images[:limit]
    
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

