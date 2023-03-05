"""Module containing data loading things."""

from pathlib import Path

import numpy as np
from hmm.types import FloatArray


def load_csv(path: Path | str) -> FloatArray:
    """Load data from txt.

    Args:
        path (Path | str): Path to CSV file.

    Returns:
        Float[Array]
    """
    file_path: Path = path if isinstance(path) else Path(path)
    return np.loadtxt(file_path, delimiter=",")
