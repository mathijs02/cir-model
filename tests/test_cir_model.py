"""
This module tests the CenteredIsotonicRegression class.
"""

import itertools
import pickle
from typing import List, Union

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


@pytest.mark.parametrize(
    "increasing, expected",
    [
        ("auto", EXAMPLE_OUTPUT_WT),
        (True, EXAMPLE_OUTPUT_WT),
        (
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
    ],
)
def test_cir_increasing(
    increasing: Union[str, bool],
    expected: List[float],
) -> None:
    """
    Test whether the expected prediction values are obtained for various values
    of `increasing`.
    """
    cir = CenteredIsotonicRegression(increasing=increasing)
    output = cir.fit_transform(EXAMPLE_X, EXAMPLE_Y, sample_weight=EXAMPLE_WT)
    np.testing.assert_array_almost_equal(expected, output)


def test_cir_fit_transform() -> None:
    """
    Test whether the same prediction values are obtained when using `fit` and
    `transform` compared to `fit_transform`.
    """
    output_f = (
        CenteredIsotonicRegression()
        .fit(EXAMPLE_X, EXAMPLE_Y, sample_weight=EXAMPLE_WT)
        .transform(EXAMPLE_X)
    )
    output_ft = CenteredIsotonicRegression().fit_transform(
        EXAMPLE_X, EXAMPLE_Y, sample_weight=EXAMPLE_WT
    )
    np.testing.assert_array_almost_equal(output_f, output_ft)


def test_cir_one_value() -> None:
    """
    Test the fitting when the training dataset contains only one datapoint.
    """
    output = CenteredIsotonicRegression(increasing="auto").fit_transform(
        [1], [5], sample_weight=[2]
    )
    np.testing.assert_array_almost_equal(output, [5])


def test_cir_no_weights() -> None:
    """
    Test the fitting when no weights are given.
    """
    output = (
        CenteredIsotonicRegression()
        .fit(EXAMPLE_X, EXAMPLE_Y)
        .transform(EXAMPLE_X)
    )
    np.testing.assert_array_almost_equal(output, EXAMPLE_OUTPUT_NO_WT)


def test_cir_no_weights_equal_ones() -> None:
    """
    Test that passing no weights gives the same output as passing ones as
    weights.
    """
    no_wt = CenteredIsotonicRegression().fit_transform(EXAMPLE_X, EXAMPLE_Y)
    wt_ones = CenteredIsotonicRegression().fit_transform(
        EXAMPLE_X, EXAMPLE_Y, sample_weight=np.ones(len(EXAMPLE_X))
    )
    np.testing.assert_array_almost_equal(no_wt, wt_ones)


@pytest.mark.parametrize("order", map(list, itertools.permutations(range(4))))
def test_cir_all_permutations(order: List[int]) -> None:
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


def test_cir_1d_array() -> None:
    """
    Test whether numpy arrays as inputs work as expected.
    """
    cir = CenteredIsotonicRegression()
    output = cir.fit_transform(
        np.array(EXAMPLE_X), np.array(EXAMPLE_Y), sample_weight=EXAMPLE_WT
    )
    np.testing.assert_array_almost_equal(EXAMPLE_OUTPUT_WT, output)


def test_cir_2d_array() -> None:
    """
    Test whether numpy arrays as input, with a 2D array for X, works as
    expected.
    """
    cir = CenteredIsotonicRegression()
    output = cir.fit_transform(
        np.array([EXAMPLE_X]).T, np.array(EXAMPLE_Y), sample_weight=EXAMPLE_WT
    )
    np.testing.assert_array_almost_equal(EXAMPLE_OUTPUT_WT, output)


def test_cir_incompatible_shapes() -> None:
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


def test_cir_oob_nan() -> None:
    """
    Test the intended behaviour of `out_of_bounds="nan"`.
    """
    cir = CenteredIsotonicRegression(out_of_bounds="nan")
    cir.fit(EXAMPLE_X, EXAMPLE_Y, EXAMPLE_WT)
    output = cir.transform(np.array([-1, 0, 1, 2]))
    expected = np.array([np.nan, np.nan, 1.0, 5.81578947])
    np.testing.assert_array_almost_equal(expected, output)


def test_cir_oob_clip() -> None:
    """
    Test the intended behaviour of `out_of_bounds="clip"`.
    """
    cir = CenteredIsotonicRegression(out_of_bounds="clip")
    cir.fit(EXAMPLE_X, EXAMPLE_Y, EXAMPLE_WT)
    output = cir.transform(np.array([-1, 0, 1, 2]))
    expected = np.array([1.0, 1.0, 1.0, 5.81578947])
    np.testing.assert_array_almost_equal(expected, output)


def test_cir_oob_raise() -> None:
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


@pytest.mark.parametrize(
    "y_min, y_max, expected",
    [
        (None, None, EXAMPLE_OUTPUT_WT),
        (0.5, 50, EXAMPLE_OUTPUT_WT),
        (
            3,
            50,
            [
                3.0,
                7.23684211,
                11.47368421,
                15.71052632,
                19.64285714,
                23.32142857,
                27.0,
                32.0,
                43.66666667,
                49.5,
            ],
        ),
        (
            25,
            50,
            [
                25.0,
                25.0,
                25.0,
                25.0,
                25.58823529,
                26.29411765,
                27.0,
                32.0,
                43.66666667,
                49.5,
            ],
        ),
        (
            3,
            None,
            [
                3.0,
                7.23684211,
                11.47368421,
                15.71052632,
                19.64285714,
                23.32142857,
                27.0,
                32.0,
                43.66666667,
                49.5,
            ],
        ),
        (
            25,
            None,
            [
                25.0,
                25.0,
                25.0,
                25.0,
                25.58823529,
                26.29411765,
                27.0,
                32.0,
                43.66666667,
                49.5,
            ],
        ),
        (
            0.5,
            45,
            [
                1.0,
                5.81578947,
                10.63157895,
                15.44736842,
                19.64285714,
                23.32142857,
                27.0,
                32.0,
                40.66666667,
                45.0,
            ],
        ),
        (
            0.5,
            25,
            [
                1.0,
                5.81578947,
                10.63157895,
                15.44736842,
                18.50097847,
                20.08610568,
                21.67123288,
                23.25636008,
                24.84148728,
                25.0,
            ],
        ),
        (
            None,
            45,
            [
                1.0,
                5.81578947,
                10.63157895,
                15.44736842,
                19.64285714,
                23.32142857,
                27.0,
                32.0,
                40.66666667,
                45.0,
            ],
        ),
        (
            None,
            25,
            [
                1.0,
                5.81578947,
                10.63157895,
                15.44736842,
                18.50097847,
                20.08610568,
                21.67123288,
                23.25636008,
                24.84148728,
                25.0,
            ],
        ),
        (
            3,
            45,
            [
                3.0,
                7.23684211,
                11.47368421,
                15.71052632,
                19.64285714,
                23.32142857,
                27.0,
                32.0,
                40.66666667,
                45.0,
            ],
        ),
        (
            22,
            35,
            [
                22.0,
                22.0,
                22.0,
                22.0,
                23.47058824,
                25.23529412,
                27.0,
                32.0,
                34.0,
                35.0,
            ],
        ),
    ],
)
def test_cir_ymin_ymax(
    y_min: float,
    y_max: float,
    expected: List[float],
) -> None:
    """
    Test whether the `y_min` and `y_max` functionality from
    `IsotonicRegression` works as expected with `CenteredIsotonicRegression`.
    """
    cir = CenteredIsotonicRegression(y_min=y_min, y_max=y_max)
    output = cir.fit_transform(EXAMPLE_X, EXAMPLE_Y, sample_weight=EXAMPLE_WT)
    np.testing.assert_array_almost_equal(expected, output)


@pytest.mark.parametrize(
    "y, non_centered_points, expected",
    [
        ([-1, -0.5, 0, 0, 0.5, 1], [0, 1], [-1.0, -0.5, 0.0, 0.0, 0.5, 1.0]),
        (
            [-1, -0.5, 0, 0, 0.5, 1],
            [1],
            [-1.0, -0.5, -0.166667, 0.166667, 0.5, 1.0],
        ),
        (
            [-1, -0.5, 0, 0, 0.5, 1],
            [],
            [-1.0, -0.5, -0.166667, 0.166667, 0.5, 1.0],
        ),
        (
            [-1, -0.5, 0.2, 0.2, 0.5, 1],
            [0.2],
            [-1.0, -0.5, 0.2, 0.2, 0.5, 1.0],
        ),
        (
            [-1, -0.5, 0.2, 0.2, 0.5, 1],
            [],
            [-1.0, -0.5, -0.03333333, 0.30000000, 0.5, 1.0],
        ),
        (
            [0, 0, 0, 2, 1.5, 2.7],
            [0],
            [0.0, 0.0, 0.0, 1.16666667, 2.06666667, 2.7],
        ),
    ],
)
def test_cir_non_centered_points(
    y: List[float],
    non_centered_points: List[float],
    expected: List[float],
) -> None:
    """
    Test whether the `non_centered_points` parameter works as expected.
    """
    x = [1, 2, 3, 4, 5, 6]
    cir = CenteredIsotonicRegression(non_centered_points=non_centered_points)
    output = cir.fit_transform(x, y)
    np.testing.assert_array_almost_equal(expected, output)


def test_cir_non_centered_points_missing() -> None:
    """
    Test whether the default value of the `non_centered_points` parameter is
    correct.
    """
    x = [1, 2, 3, 4, 5, 6]
    y = [-1, -0.5, 0, 0, 0.5, 1]
    cir_missing = CenteredIsotonicRegression().fit_transform(x, y)
    cir_default = CenteredIsotonicRegression(
        non_centered_points=[0, 1]
    ).fit_transform(x, y)
    np.testing.assert_array_almost_equal(cir_missing, cir_default)
