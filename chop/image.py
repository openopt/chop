"""Utils functions for image manipulation and visualization"""

import matplotlib.pyplot as plt
import numpy as np


def matplotlib_imshow(img, one_channel=False, ax=None):
    if ax is None:
        ax = plt
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.detach().cpu().numpy()
    if one_channel:
        ax.imshow(npimg, cmap="gray")
    else:
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
