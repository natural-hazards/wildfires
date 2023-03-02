import numpy as np


class TransformFFT(object):

    def __init__(self,
                 nfeatures: int = None):

        self._nfeatures = None
        self.nfeatures = nfeatures

    @property
    def nfeatures(self) -> int:

        return self._nfeatures

    @nfeatures.setter
    def nfeatures(self, n) -> None:

        self._nfeatures = n

    def transform(self, samples: np.ndarray) -> np.ndarray:

        frequency_dom = np.abs(np.fft.fft(samples, self._nfeatures))
        return frequency_dom
