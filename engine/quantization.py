"""Scalar quantization: float32 <-> int8 with column-wise scaling."""

from __future__ import annotations

import numpy as np


def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float32 vectors to int8, returning quantized vectors and scale factors.

    Each column (dimension) is scaled by its max absolute value.
    Returns (int8_vectors, scales) where scales shape is (dimensions,).
    """
    # Compute per-column scale: max absolute value, protect against zero
    abs_vectors = np.abs(vectors)
    scales = np.max(abs_vectors, axis=0, keepdims=True).astype(np.float32)  # (1, D)
    scales = np.maximum(scales, 1e-8)  # avoid div-by-zero

    # Scale, round to int8, clip
    scaled = vectors / scales  # (N, D) in approximately [-128, 127] range
    int8_vectors = np.round(scaled).astype(np.int8)
    # Clip to int8 range just in case
    int8_vectors = np.clip(int8_vectors, np.iinfo(np.int8).min, np.iinfo(np.int8).max)

    return int8_vectors, scales.squeeze(0)  # scales shape (D,)


def dequantize_int8(int8_vectors: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize int8 vectors back to float32 using the given scales."""
    scales_reshaped = scales.astype(np.float32).reshape(1, -1)
    return (int8_vectors.astype(np.float32) * scales_reshaped)