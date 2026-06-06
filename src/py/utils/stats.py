# stats.py
"""Shared statistical helpers for MOSAIC pipeline scripts."""

import numpy as np
from scipy.stats import fisher_exact


def fisher_enrichment(
    in_cluster: np.ndarray, has_feature: np.ndarray
) -> tuple[float, float]:
    """Two-sided Fisher's exact test for cluster-feature enrichment.

    Args:
        in_cluster: Boolean array indicating cluster membership.
        has_feature: Boolean array indicating feature presence.

    Returns:
        Tuple of (odds_ratio, p_value).
    """
    table = np.array(
        [
            [np.sum(in_cluster & has_feature), np.sum(in_cluster & ~has_feature)],
            [np.sum(~in_cluster & has_feature), np.sum(~in_cluster & ~has_feature)],
        ]
    )
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    return float(odds_ratio), float(p_value)


def bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply Bonferroni correction to a list of p-values.

    Args:
        p_values: Raw p-values.

    Returns:
        Corrected p-values clipped to [0, 1].
    """
    n_tests = len(p_values)
    return [min(p * n_tests, 1.0) for p in p_values]
