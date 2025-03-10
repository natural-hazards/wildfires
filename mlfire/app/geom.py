from geographiclib.geodesic import Geodesic


class RectangleArea(object):

    def __init__(self, center=None, width: float = None, height: float = None, project_to_earth: bool = None):

        self.center = None
        if center is not None: self.center = center

        self._width = None
        if width is not None: self.width = width

        self._height = None
        if height is not None: self.height = height

        self._project_to_earth = False
        if project_to_earth is not None: self._project_to_earth = project_to_earth

    @property
    def center(self):

        return self._center

    @center.setter
    def center(self, pos_center) -> None:

        self._center = pos_center

    @property
    def width(self) -> float:

        return self._width

    @width.setter
    def width(self, value: float) -> None:

        if value <= 0:
            raise ValueError('Width value must be positive integer!')

        self._width = value

    @property
    def height(self) -> float:

        return self._height

    @height.setter
    def height(self, value: float) -> None:

        if value <= 0:
            raise ValueError('Height value must be positive integer!')

        self._height = value

    @property
    def projectToEarth(self) -> bool:

        return self._project_to_earth

    @projectToEarth.setter
    def projectToEarth(self, value: bool):

        self._project_to_earth = value

    def __boundsRectangle(self) -> list:

        proj_geod = Geodesic.WGS84  # folium takes coordinates in EPSG:4326, i.e. WSG84

        a = self.width / 2.
        b = self.height / 2.

        lat = self.center[0]
        lon = self.center[1]

        right = proj_geod.Direct(lat, lon, 90, a)
        left = proj_geod.Direct(lat, lon, -90, a)

        top = proj_geod.Direct(lat, lon, 0, b)
        bottom = proj_geod.Direct(lat, lon, 180, b)

        return [[top['lat2'], left['lon2']],
                [bottom['lat2'], right['lon2']]]

    @property
    def bounds(self) -> list:

        return self.__boundsRectangle()
