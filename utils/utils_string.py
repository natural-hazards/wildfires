import string
import random


def getRandomString(length: int) -> str:

    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))

    return result_str
