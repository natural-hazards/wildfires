import json

from pathlib import Path as OSPath

from matplotlib import colors as plt_colors
from matplotlib import cm as plt_cmap
from matplotlib.colors import rgb2hex


class CMapHelper(object):

    def __init__(self, lst_colors: list[str], vmin: int, vmax: int) -> None:

        # ListedColormap takes a string related to color specification in as #XXX,
        # where X is number in hexadecimal format
        _lst_colors = ['#{}'.format(colour_hex) if colour_hex[0] != '#' else colour_hex for colour_hex in lst_colors]

        self._norm = plt_colors.Normalize(vmin=vmin, vmax=vmax)
        self._cmap = plt_colors.LinearSegmentedColormap.from_list('CMap', colors=_lst_colors)
        self._scalar_map = plt_cmap.ScalarMappable(norm=self._norm, cmap=self._cmap)

        self._vmin = vmin
        self._vmax = vmax

    @property
    def vmin(self) -> int:

        return self._vmin

    @property
    def vmax(self) -> int:

        return self._vmax

    def getRGB(self, val: int):

        rbg = self._scalar_map.to_rgba(val)[0:-1]

        return rbg

    def save(self, filename: OSPath):

        n = self.vmax + 1
        lst_colors = [rgb2hex(self._scalar_map.to_rgba(v)[0:-1]) for v in range(n)]

        with open(filename, 'w') as f:

            cmap_dict = {
                'colors': lst_colors,
                'values': {'vmin': self._vmin, 'vmax': self._vmax}
            }

            json.dump(cmap_dict, f)
