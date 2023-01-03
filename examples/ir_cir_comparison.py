"""
This module shows an example usage of `CenteredIsotonicRegression` and saves a
plot comparing IR and CIR models fitted on the same data.
"""

import matplotlib.pyplot as plt
from benchmark import generate_data
from sklearn.isotonic import IsotonicRegression

from cir_model import CenteredIsotonicRegression

x, y = generate_data(100)
ir = IsotonicRegression(increasing=False).fit(x, y)
cir = CenteredIsotonicRegression(increasing=False).fit(x, y)

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.scatter(x, y, color="C2", alpha=0.2, label="data")
ax.plot(
    ir.f_.x,
    ir.f_.y,
    color="C0",
    marker="o",
    markersize=5,
    label="IR",
    alpha=0.6,
)
ax.plot(
    cir.f_.x,
    cir.f_.y,
    color="C3",
    marker="o",
    markersize=5,
    label="CIR",
    alpha=0.6,
)
ax.legend()

plt.savefig("ir_cir_comparison.png")
