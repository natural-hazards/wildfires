from mlfire.utils.functool import lazy_import

_np = lazy_import('matplotlib.pyplot')


def imshow(src: _np.ndarray,
           ax=None,
           title: str = None,
           figsize: tuple = None,
           tight_layout: bool = True,
           show: bool = False) -> None:

    plt = lazy_import('matplotlib.pyplot')

    if ax is not None:

        ax.imshow(src)
        ax.axis('off')
        if title is not None: ax.set_title(title)

    else:

        if figsize is not None: plt.rcParams['figure.figsize'] = figsize
        plt.imshow(src)
        plt.axis('off')
        if title is not None: plt.title(title)

    if tight_layout:
        plt.tight_layout()

    if show:
        plt.show()

# TODO show labels
