"""
This module tests the CenteredIsotonicRegression class.
"""

import itertools
import pickle
from typing import List, Optional, Union

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cir_model import CenteredIsotonicRegression

EXAMPLE_X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
EXAMPLE_Y = [1, 37, 42, 5, 7, 14, 27, 32, 56, 43]
EXAMPLE_WT = [1, 1, 2, 2, 3, 3, 1, 1, 4, 4]
EXAMPLE_OUTPUT_WT = [
    1.0,
    5.81578947,
    10.63157895,
    15.44736842,
    19.64285714,
    23.32142857,
    27.0,
    32.0,
    43.66666667,
    49.5,
]
EXAMPLE_OUTPUT_NO_WT = [
    1.0,
    7.66666667,
    14.33333333,
    21.0,
    23.0,
    25.0,
    27.0,
    32.0,
    43.66666667,
    49.5,
]

EXAMPLES = [
    # auto increasing/decreasing
    (
        EXAMPLE_X,
        EXAMPLE_Y,
        EXAMPLE_WT,
        "auto",
        EXAMPLE_OUTPUT_WT,
    ),
    # manual increasing
    (
        EXAMPLE_X,
        EXAMPLE_Y,
        EXAMPLE_WT,
        True,
        EXAMPLE_OUTPUT_WT,
    ),
    # manual decreasing
    (
        EXAMPLE_X,
        EXAMPLE_Y,
        EXAMPLE_WT,
        False,
        [
            30.5,
            30.5,
            30.32786885,
            30.09836066,
            29.86885246,
            29.63934426,
            29.40983607,
            29.33333333,
            29.33333333,
            29.33333333,
        ],
    ),
    # no weights
    (
        EXAMPLE_X,
        EXAMPLE_Y,
        None,
        "auto",
        EXAMPLE_OUTPUT_NO_WT,
    ),
    # consecutive zeros in middle of range
    (
        [1, 2, 3, 4, 5, 6],
        [-1, -0.5, 0, 0, 0.5, 1],
        None,
        "auto",
        [-1.0, -0.5, 0.0, 0.0, 0.5, 1.0],
    ),
    # consecutive non-zeros in middle of range
    (
        [1, 2, 3, 4, 5, 6],
        [-1, -0.5, 0.2, 0.2, 0.5, 1],
        None,
        "auto",
        [-1.0, -0.5, -0.03333333, 0.3, 0.5, 1.0],
    ),
    # one datapoint
    ([1], [5], [2], "auto", [5]),
    # starting with multiple zeros
    (
        [1, 2, 3, 4, 5],
        [0, 0, 0, 2, 1.5],
        None,
        "auto",
        [0.0, 0.0, 0.0, 1.16666667, 1.75],
    ),
]


@pytest.mark.parametrize("x, y, sample_weight, increasing, expected", EXAMPLES)
def test_cir_fit(
    x: List[float],
    y: List[float],
    sample_weight: Optional[List[float]],
    increasing: Union[str, bool],
    expected: List[float],
) -> None:
    """
    Test whether the expected prediction values are obtained when using `fit`
    and `transform`.
    """
    cir = CenteredIsotonicRegression(increasing=increasing)
    cir.fit(x, y, sample_weight)
    output = cir.transform(x)
    np.testing.assert_array_almost_equal(expected, output)


@pytest.mark.parametrize("x, y, sample_weight, increasing, expected", EXAMPLES)
def test_cir_fit_transform(
    x: List[float],
    y: List[float],
    sample_weight: Optional[List[float]],
    increasing: Union[str, bool],
    expected: List[float],
) -> None:
    """
    Test whether the expected prediction values are obtained when using
    `fit_transform`.
    """
    cir = CenteredIsotonicRegression(increasing=increasing)
    output = cir.fit_transform(x, y, sample_weight=sample_weight)
    np.testing.assert_array_almost_equal(expected, output)


@pytest.mark.parametrize("order", map(list, itertools.permutations(range(4))))
def test_cir_fit_all_permutations(order: List[int]) -> None:
    """
    Test whether the outcome is independent of the ordering in the training
    dataset by testing all possible permutations of a small dataset.
    """
    print(order)
    x = np.array([1, 2, 3, 4])[order]
    y = np.array([1, 37, 42, 5])[order]
    wt = np.array([1, 1, 2, 2])[order]

    cir = CenteredIsotonicRegression()
    cir.fit(x, y, wt)

    output = cir.transform([1, 2, 3, 4])
    np.testing.assert_array_almost_equal(
        [1.0, 12.45454545, 23.90909091, 26.2], output
    )


@pytest.mark.parametrize("x, y, sample_weight, increasing, expected", EXAMPLES)
def test_cir_fit_1d_array(
    x: List[float],
    y: List[float],
    sample_weight: Optional[List[float]],
    increasing: Union[str, bool],
    expected: List[float],
) -> None:
    """
    Test whether numpy arrays as inputs work as expected.
    """
    wt = None if sample_weight is None else np.array(sample_weight)
    cir = CenteredIsotonicRegression(increasing=increasing)
    output = cir.fit_transform(np.array(x), np.array(y), sample_weight=wt)
    np.testing.assert_array_almost_equal(expected, output)


@pytest.mark.parametrize("x, y, sample_weight, increasing, expected", EXAMPLES)
def test_cir_fit_2d_array(
    x: List[float],
    y: List[float],
    sample_weight: Optional[List[float]],
    increasing: Union[str, bool],
    expected: List[float],
) -> None:
    """
    Test whether numpy arrays as input, with a 2D array for X, works as
    expected.
    """
    wt = None if sample_weight is None else np.array(sample_weight)
    cir = CenteredIsotonicRegression(increasing=increasing)
    output = cir.fit_transform(np.array([x]).T, np.array(y), sample_weight=wt)
    np.testing.assert_array_almost_equal(expected, output)


def test_cir_fit_no_weights() -> None:
    """
    Test that passing no weights gives the same output as passing ones as
    weights.
    """
    no_wt = CenteredIsotonicRegression().fit_transform(EXAMPLE_X, EXAMPLE_Y)
    wt_ones = CenteredIsotonicRegression().fit_transform(
        EXAMPLE_X, EXAMPLE_Y, sample_weight=np.ones(len(EXAMPLE_X))
    )
    np.testing.assert_array_almost_equal(no_wt, wt_ones)


def test_cir_fit_incompatible_shapes() -> None:
    """
    Test whether an error is thrown when incompatible array sizes are passed.
    """
    cir = CenteredIsotonicRegression()
    with pytest.raises(ValueError):
        cir.fit(EXAMPLE_X, EXAMPLE_Y, EXAMPLE_WT[:-2])


def test_cir_pickle() -> None:
    """
    Test whether pickling and unpickling a model results in the same
    predictions.
    """
    cir = CenteredIsotonicRegression()
    cir.fit(EXAMPLE_X, EXAMPLE_Y, EXAMPLE_WT)
    original_output = cir.transform(EXAMPLE_X)

    cir_pickled = pickle.dumps(cir)
    cir_unpickled = pickle.loads(cir_pickled)
    pickled_output = cir_unpickled.transform(EXAMPLE_X)

    np.testing.assert_array_almost_equal(original_output, pickled_output)


def test_cir_fit_transform_oob_nan() -> None:
    """
    Test the intended behaviour of `out_of_bounds="nan"`.
    """
    cir = CenteredIsotonicRegression(out_of_bounds="nan")
    cir.fit(EXAMPLE_X, EXAMPLE_Y, EXAMPLE_WT)
    output = cir.transform(np.array([-1, 0, 1, 2]))
    expected = np.array([np.nan, np.nan, 1.0, 5.81578947])
    np.testing.assert_array_almost_equal(expected, output)


def test_cir_fit_transform_oob_clip() -> None:
    """
    Test the intended behaviour of `out_of_bounds="clip"`.
    """
    cir = CenteredIsotonicRegression(out_of_bounds="clip")
    cir.fit(EXAMPLE_X, EXAMPLE_Y, EXAMPLE_WT)
    output = cir.transform(np.array([-1, 0, 1, 2]))
    expected = np.array([1.0, 1.0, 1.0, 5.81578947])
    np.testing.assert_array_almost_equal(expected, output)


def test_cir_fit_transform_oob_raise() -> None:
    """
    Test the intended behaviour of `out_of_bounds="raise"`.
    """
    cir = CenteredIsotonicRegression(out_of_bounds="raise")
    cir.fit(EXAMPLE_X, EXAMPLE_Y, EXAMPLE_WT)
    with pytest.raises(ValueError):
        cir.transform(np.array([-1, 0, 1, 2]))


def test_cir_pipeline() -> None:
    """
    Test whether CenteredIsotonicRegression is compatible with scikit-learn
    pipelines.
    """
    x = np.array([EXAMPLE_X]).T
    y = np.array(EXAMPLE_Y)
    wt = np.array(EXAMPLE_WT)

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("cir", CenteredIsotonicRegression())]
    )
    pipe.fit(x, y, cir__sample_weight=wt)
    output = pipe.transform(x)
    np.testing.assert_array_almost_equal(EXAMPLE_OUTPUT_WT, output)
