import re

from PyQt5.QtGui import QValidator


class ValidLatitude(QValidator):

    def __init__(self, parent=None):

        super().__init__(parent)

    def validate(self, s, pos):

        if s == '':
            return QValidator.Acceptable, s, pos

        re_float = r'[-]?([0-9]+([.][0-9]*)?|[.][0-9]+)'
        re_pattern = r'{}\,\s*{}'.format(re_float, re_float)

        if re.fullmatch('-', s):
            return QValidator.Intermediate, s, pos
        elif re.fullmatch(re_pattern, s):
            lat_lon = s.split(',')

            lat = float(lat_lon[0])
            lon = float(lat_lon[1])

            if float(lat) < -180. or float(lat) > 180.:
                return QValidator.Invalid, s, pos

            if float(lon) < -180. or float(lon) > 180.:
                return QValidator.Invalid, s,  pos

            return QValidator.Acceptable, s, pos
        elif re.fullmatch(re_float, s):
            lat = float(s)

            if -180 <= lat <= 180.:
                return QValidator.Acceptable, s, pos
            else:
                return QValidator.Invalid, s, pos
        else:
            return QValidator.Invalid, s, pos

    def fixup(self, s):

        pass
