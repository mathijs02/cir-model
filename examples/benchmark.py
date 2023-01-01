"""
This module runs a benchmark of IsotonicRegression and
CenteredIsotonicRegression, to understand how they are different in regards to
computational resources.
"""

import logging
import time
from datetime import timedelta
from typing import Tuple

import numpy as np
from scipy.stats import logistic
from sklearn.isotonic import IsotonicRegression

from cir_model import CenteredIsotonicRegression

logging.basicConfig(level=logging.INFO)


def generate_data(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an artifical dataset with a noisy sigmoid data generating
    function.
    """
    np.random.seed(10)
    x = np.sort(np.random.uniform(0, 35, n))
    noise = np.random.normal(0, 2, n)
    y = logistic.cdf(-x / 3 + 5 + noise)
    return x, y


def benchmark_ir_cir(n_datapoints: int, n_repeats: int, n_cycles: int) -> None:
    """
    Run a benchmark of CenteredIsotonicRegression against IsotonicRegression.
    The model fitting is repeated, to obtain less noisy estimates, and it
    cycles multiple times through the models to obtain a fair comparison: the
    first run is typically slower.
    """
    x, y = generate_data(n_datapoints)

    for model in n_cycles * [IsotonicRegression, CenteredIsotonicRegression]:
        start_time = time.monotonic()
        for _ in range(n_repeats):
            model().fit_transform(x, y)

        end_time = time.monotonic()
        time_diff = timedelta(seconds=end_time - start_time)
        logging.info(f"Time taken by {model}: {time_diff}")


if __name__ == "__main__":
    benchmark_ir_cir(100000, 500, 3)
