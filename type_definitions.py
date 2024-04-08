from typing import Union, Tuple
from enum import Enum

import numpy as np
import pandas as pd

Vectorizable = Union[float, int, np.ndarray, pd.Series, pd.DataFrame]
Vectorizable1d = Union[np.ndarray, pd.Series]
Vectorizable2d = Union[np.ndarray, pd.DataFrame]

Parameters = Tuple[float, ...]