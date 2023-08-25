import datetime
import string
import random
import re


def getRandomString(length: int) -> str:

    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))

    return result_str


def band2date_reflectance(band_name) -> datetime.datetime.date:  # TODO rename -> dsc2date_reflectance

    str_date = re.match(r'\d{4}_\d{2}_\d{2}', band_name).group(0)
    return datetime.datetime.strptime(str_date, '%Y_%m_%d').date()


def band2date_tempsurface(band_name) -> datetime.datetime.date:

    str_date = re.match(r'\d{4}_\d{2}_\d{2}', band_name).group(0)
    return datetime.datetime.strptime(str_date, '%Y_%m_%d').date()


def band2date_firecci(band_name) -> datetime.datetime.date:

    str_date = re.match(r'\d{4}_\d{2}_\d{2}', band_name).group(0)
    return datetime.datetime.strptime(str_date, '%Y_%m_%d').date()


def band2date_mtbs(band_name) -> datetime.datetime.date:

    str_date = re.search(r'\d{4}', band_name).group(0)
    return datetime.datetime.strptime(str_date, '%Y').date()
