"""Module containing data loading things."""

from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_csv(path: Path | str) -> npt.NDArray[np.float_]:
    """Load data from txt.

    Args:
        path (Path | str): Path to CSV file.

    Returns:
        Float[Array]
    """
    file_path: Path = path if isinstance(path) else Path(path)
    return np.loadtxt(file_path, delimiter=",")
