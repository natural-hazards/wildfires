from mlfire.utils.functool import lazy_import

# lazy imports
_colors = lazy_import('mlfire.utils.colors')
_cmap = lazy_import('mlfire.utils.cmap')
_np = lazy_import('numpy')
_plt = lazy_import('matplotlib.pyplot')

def imshow(src: _np.ndarray,
           ax=None,
           title: str = None,
           figsize: tuple = None,
           tight_layout: bool = True,
           show: bool = False,
           interpolation: str = 'antialiased') -> None:

    if ax is not None:
        ax.imshow(src, interpolation=interpolation)
        ax.axis('off')
        if title is not None: ax.set_title(title)
    else:
        if figsize is not None: _plt.rcParams['figure.figsize'] = figsize

        _plt.imshow(src, interpolation=interpolation)
        _plt.axis('off')
        if title is not None: _plt.title(title)

    if tight_layout: _plt.tight_layout()
    if show: _plt.show()


def labelshow(labels: _np.ndarray,
              with_uncharted_areas: bool = False,
              ax=None,
              title: str = None,
              figsize: tuple = None,
              tight_layout: bool = True,
              show: bool = False) -> None:

    label_rendered = _np.empty(shape=labels.shape + (3,), dtype=_np.uint8)

    label_rendered[:] = _colors.Colors.GRAY_COLOR.value
    label_rendered[labels == 1, :] = _colors.Colors.RED_COLOR.value

    # show uncharted pixels
    if with_uncharted_areas: label_rendered[_np.isnan(labels), :] = 0

    imshow(label_rendered, ax=ax, title=title, figsize=figsize, tight_layout=tight_layout, show=show)


def labelsave(labels: _np.ndarray,
              with_uncharted_areas: bool = False,
              fn: str = 'image.png') -> None:

    label_rendered = _np.empty(shape=labels.shape + (3,), dtype=_np.uint8)

    label_rendered[:] = _colors.Colors.GRAY_COLOR.value
    label_rendered[labels == 1, :] = _colors.Colors.RED_COLOR.value

    # show uncharted pixels
    if with_uncharted_areas: label_rendered[_np.isnan(labels), :] = 0

    _plt.imsave(fn, label_rendered)


def labelshow_prob(labels: _np.ndarray,
                   with_uncharted_areas: bool = False,
                   ax=None,
                   title: str = None,
                   figsize: tuple = None,
                   tight_layout: bool = True,
                   show: bool = False) -> None:

    cmap = lazy_import('mlfire.utils.cmap')
    lst_colors = ['#5A5A55', '#FFF']

    cmap_helper = cmap.CMapHelper(lst_colors=lst_colors, vmin=0, vmax=1)

    label_rendered = _np.uint8(cmap_helper.getRGBA(labels)[:, :, :-1] * 255)
    if with_uncharted_areas: label_rendered[_np.isnan(labels), :] = 0

    imshow(label_rendered, ax=ax, title=title, figsize=figsize, tight_layout=tight_layout, show=show)
