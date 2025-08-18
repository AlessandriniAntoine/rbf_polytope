from dataclasses import dataclass
import numpy as np


@dataclass
class RbfData:
    radius: float 
    center: np.ndarray

@dataclass
class RbfModel:
    centers: np.ndarray
    widths: np.ndarray
    vertices: np.ndarray
    center_norm: np.ndarray
    is_continuous: bool

    def __post_init__(self):
        assert (self.centers.shape[0] == self.widths.shape[0])