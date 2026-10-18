"""Scientific statistics implementation extracted without changing calculation order."""

import numpy as np


def holm_bonferroni(p_values):
    """
    Manual Holm-Bonferroni correction.

    Returns adjusted p-values in the original order.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m)

    for i, p in enumerate(sorted_p):
        adjusted_sorted[i] = (m - i) * p

    # monotonicity correction
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    return adjusted
