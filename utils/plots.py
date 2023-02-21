import cv2 as opencv
import numpy as np

import matplotlib.pyplot as plt


def imshow(src: np.ndarray,
           ax=None,
           title: str = None,
           figsize: tuple = None,
           to_bgra: bool = True,
           tight_layout: bool = True,
           show: bool = False) -> None:

    # matplot lib works with rgb order of channel so input image needs conversion
    src_rgb = opencv.cvtColor(src, opencv.COLOR_BGR2RGB) if to_bgra and len(src.shape) == 3 else src

    if ax is not None:
        ax.imshow(src_rgb)
        ax.axis('off')
        if title is not None: ax.set_title(title)
    else:
        if figsize is not None: plt.rcParams['figure.figsize'] = figsize
        plt.imshow(src_rgb)
        plt.axis('off')
        if title is not None: plt.title(title)

    if tight_layout:
        plt.tight_layout()

    if show:
        plt.show()
