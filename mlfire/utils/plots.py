from mlfire.utils.functool import lazy_import

_np = lazy_import('numpy')


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


def labelshow(labels: _np.ndarray,
              with_uncharted_areas: bool = False,
              ax=None,
              title: str = None,
              figsize: tuple = None,
              tight_layout: bool = True,
              show: bool = False) -> None:

    # lazy imports
    colors = lazy_import('mlfire.utils.colors')

    label_rendered = _np.empty(shape=labels.shape + (3,), dtype=_np.uint8)

    label_rendered[:] = colors.Colors.GRAY_COLOR.value
    label_rendered[labels == 1, :] = colors.Colors.RED_COLOR.value

    # show uncharted pixels
    if with_uncharted_areas: label_rendered[_np.isnan(labels), :] = 0

    imshow(label_rendered, ax=ax, title=title, figsize=figsize, tight_layout=tight_layout, show=show)
