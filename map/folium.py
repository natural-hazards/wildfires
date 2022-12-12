import folium

from folium import plugins as _FoliumPlugins
from map.plugins import draw as _MapPluginDraw

from IPython.display import display_html


class FoliumMap(object):

    def __init__(self, location, shape, zoom_start) -> None:

        self._location = location

        self._zoom_start = zoom_start
        self._shape = shape

        self._map = None

    @property
    def map(self) -> folium.Map:

        if self._map is None:
            self.__initMap()

        return self._map

    def __initMap(self) -> None:

        self._map = folium.Map(location=self._location,
                               zoom_start=self._zoom_start,
                               control_scale=True,
                               tiles='cartodbdark_matter')

        # set draw plugin
        opt_draw = {'circle': False,
                    'circlemarker': False,
                    'polyline': False,
                    'polygon': False,
                    'rectangle': True,
                    'marker': True}
        plugin_draw = _MapPluginDraw.Draw(export=True,
                                          draw_options=opt_draw,
                                          edit_options=None)
        self._map.add_child(plugin_draw)

        # add minimap
        mini_map = _FoliumPlugins.MiniMap()
        self._map.add_child(mini_map)

        # add mouse position
        roundnum = "function(num) {return L.Util.formatNum(num, 5);};"
        mouse_position = _FoliumPlugins.MousePosition(position='topright',
                                                      separator=' | ',
                                                      prefix="Position:",
                                                      lat_formatter=roundnum,
                                                      lng_formatter=roundnum)
        self._map.add_child(mouse_position)

    def show(self) -> None:

        map = self.map
        display_html(map)


