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


def matplotlib_imshow_batch(batch, labels=None, one_channel=False, axes=None, normalize=False, range=(0., 1.),
                            title=""):
    npimgs = [img.detach().cpu().numpy() for img in batch]
    if labels is None:
        labels = [""] * batch.size(0)
    axes[0].set_ylabel(title, rotation=0)
    for ax, img, label in zip(axes, npimgs, labels):
        if one_channel:
            if normalize:
                img = normalize_image(img, range, True)
            ax.imshow(img, cmap='gray')
        else:
            img = np.transpose(img, (1, 2, 0))
            if normalize:
                img = normalize_image(img, range, False)
            ax.imshow(img)

        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis('off')



def normalize_image(img, range=(0., 1.), one_channel=False):
    """
    Linearly normalizes an image to be in range.

    Supposes that img is in numpy image shape: (m, n, channels) or (m, n)
    """
    new_min, new_max = range
    old_min, old_max = img.min(), img.max()
    return new_min + (img - old_min) * (new_max - new_min) / (old_max - old_min)
