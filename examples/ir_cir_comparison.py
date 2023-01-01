"""
This module shows an example usage of `CenteredIsotonicRegression` and saves a
plot comparing IR and CIR models fitted on the same data.
"""

import matplotlib.pyplot as plt
from benchmark import generate_data
from sklearn.isotonic import IsotonicRegression

from cir_model import CenteredIsotonicRegression

x, y = generate_data(100)
y_ir = IsotonicRegression(increasing=False).fit_transform(x, y)
y_cir = CenteredIsotonicRegression(increasing=False).fit_transform(x, y)

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.scatter(x, y, color="C2", alpha=0.2, label="data")
ax.plot(x, y_ir, color="C0", label="IR")
ax.plot(x, y_cir, color="C3", label="CIR")
ax.legend()

plt.savefig("ir_cir_comparison.png")
