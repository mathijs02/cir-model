[![](https://img.shields.io/pypi/v/cir-model)](https://pypi.org/project/cir-model/)

# Introduction
The `cir-model` library implements the *Centered Isotonic Regression* (CIR) model.[1] CIR is a variant of [Isotonic Regression](https://en.wikipedia.org/wiki/Isotonic_regression) (IR), which is a nonparametric regression model which only assumes that the data generating function is monotonically increasing or decreasing. The function can take any monotonic form that minimizes the sum of squared errors. [`scikit-learn`](https://scikit-learn.org/stable/) has implemented Isotonic Regression as [`IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html).[2]

In some scenarios, the assumption of monotonicity is too weak. IR can result in a function that has constant, non-decreasing intervals. In some applications, like dose-response curve fitting or in cumululative distribution function estimation, it can be safe to make the additional assumption of *strict monotonicity*: at any point the function should be increasing, not just non-decreasing (or vice versa: decreasing, not just non-increasing). This additional requirement results in a function with the desirable property that the inverse function is a unique mapping, because there are no longer ranges of constant intervals.

*Centered Isotonic Regression* (CIR) is similar to IR, but with the additional assumption of strict monotonicity. The algorithm is described in detail in [1]. The plot below shows a comparison of IR (in blue) and CIR (in red) fitted to the same datapoints (in green).

![](https://github.com/mathijs02/cir-model/raw/main/examples/ir_cir_comparison.png)

This Python library, `cir-model`, implements `CenteredIsotonicRegression` in line with the algorithm in [1]. `cir-model` uses the `IsotonicRegression` implementation of `scikit-learn`. Therefore, `CenteredIsotonicRegression` takes the same parameters as `IsotonicRegression` and is completely compatible with `scikit-learn`. You can use it for example in `scikit-learn` pipelines.

You can install `cir-model` using `pip`:
```
pip install -U cir-model
```

# Examples
To fit a CIR model:
```
>>> from cir_model import CenteredIsotonicRegression
>>> x = [1, 2, 3, 4]
>>> y = [1, 37, 42, 5]
>>> model = CenteredIsotonicRegression().fit(x, y)
>>> model.transform(x)
array([ 1. , 14.5, 28. , 28. ])
```

Finding the inverse of the CIR model above, for example for the value `x` for which `y=25`:
```
>>> from scipy import optimize
>>> optimize.newton(lambda x: model.transform([x]) - 25, 2)
array([2.77777778])
```

# References
1. [Centered Isotonic Regression: Point and Interval Estimation for Dose-Response Studies](https://arxiv.org/abs/1701.05964), Assaf P. Oron & Nancy Flournoy, Statistics in Biopharmaceutical Research, Volume 9, Issue 3, 258-267, 2017
2. [Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Fabian Pedregosa et al., JMLR 12, 2825-2830, 2011