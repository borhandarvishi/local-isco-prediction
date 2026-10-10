"""Matryoshka shortening: keep the first N values, then L2-normalize."""

from __future__ import annotations

import numpy as np


def fit_embedding(values: list[float], dimension: int) -> tuple[list[float], int]:
    """Keep the first `dimension` values, then L2-normalize.

    Truncation has to happen before normalization, including when the provider
    already normalized the full vector.
    """
    incoming = len(values)
    if incoming < dimension:
        raise ValueError(f"provider returned {incoming} dimensions, need at least {dimension}")
    array = np.asarray(values[:dimension], dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("embedding contains NaN or infinity")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("embedding norm is zero")
    unit = (array / norm).tolist()
    return unit, incoming
