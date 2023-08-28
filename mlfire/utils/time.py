from enum import Enum
from timeit import default_timer


class TimePeriod(Enum):

    YEARS = 0
    MONTHS = 1


class elapsed_timer(object):

    def __init__(self, msg: str, enable: bool = True):  # TODO rename measure

        self._msg = msg
        self._measure = enable

    def __enter__(self):

        if not self._measure:
            return

        print('Start event: \'{}\'.'.format(self._msg), flush=True)
        self._start = default_timer()

    def __exit__(self,
                 ex_type,
                 val,
                 traceback):

        if not self._measure:
            return

        self._end = default_timer()
        print('Finnish event: \'{}\'. It takes {:.2f}s.'.format(self._msg, self._end - self._start), flush=True)
