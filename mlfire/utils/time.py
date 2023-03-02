from enum import Enum
from timeit import default_timer


class TimePeriod(Enum):

    YEARS = 0
    MONTHS = 1


class elapsed_timer(object):

    def __init__(self,
                 title: str):

        self._title = title

    def __enter__(self):

        print('Start event: \'{}\'.'.format(self._title), flush=True)
        self._start = default_timer()

    def __exit__(self,
                 ex_type,
                 val,
                 traceback):

        self._end = default_timer()
        print('Finnish event: \'{}\'. It takes {:.2f}s.'.format(self._title, self._end - self._start), flush=True)
