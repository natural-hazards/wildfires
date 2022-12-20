
import datetime
import string
import random
import re


def getRandomString(length: int) -> str:

    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))

    return result_str


def band2date(band_name) -> datetime.datetime.date:

    str_date = re.match(r'\d{4}_\d{2}_\d{2}', band_name).group(0)

    return datetime.datetime.strptime(str_date, '%Y_%m_%d').strftime('%Y-%m-%d')
