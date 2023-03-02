import ee as earthengine
import folium

from folium import plugins as _FoliumPlugins
from folium.plugins import MousePosition as _FoliumMousePosition

from mlfire.app.map.plugins import draw as _MapPluginDraw

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

        def addGoogleEarthEngineLayer(self_map, obj_earthengine, vis_params, name, show: bool = False) -> None:

            if isinstance(obj_earthengine, earthengine.Image):
                map_id_dict = earthengine.Image(obj_earthengine).getMapId(vis_params)

                folium.raster_layers.TileLayer(
                    tiles=map_id_dict['tile_fetcher'].url_format,
                    attr='Map Data © Google Earth Engine',
                    name=name,
                    overlay=True,
                    control=True,
                    show=show
                ).add_to(self_map)
            else:
                raise RuntimeError('Not supported layer object')

        folium.Map.addGoogleEarthEngineLayer = addGoogleEarthEngineLayer

        # create folium map
        self._map = folium.Map(location=self._location,
                               zoom_start=self._zoom_start,
                               control_scale=True,
                               tiles='cartodbdark_matter')

        # set draw plugin
        opt_draw = {
            'circle': False,
            'circlemarker': False,
            'polyline': False,
            'polygon': False,
            'rectangle': True,
            'marker': True
        }

        plugin_draw = _MapPluginDraw.Draw(export=True, draw_options=opt_draw, edit_options=None)
        self._map.add_child(plugin_draw)

        # add minimap
        mini_map = _FoliumPlugins.MiniMap()
        self._map.add_child(mini_map)

        # add mouse position handler
        roundnum = "function(num) {return L.Util.formatNum(num, 5);};"
        mouse_position = _FoliumMousePosition(
            position='topright',
            separator=' | ',
            prefix="Position:",
            lat_formatter=roundnum,
            lng_formatter=roundnum
        )
        self._map.add_child(mouse_position)

    def addGoogleEarthEngineLayer(self, obj, params, name, show: bool = False) -> None:

        self._map.addGoogleEarthEngineLayer(obj, params, name, show)

    def show(self) -> None:

        map = self.map
        display_html(map)
