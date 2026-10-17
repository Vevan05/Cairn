"""Scalar quantization: float32 <-> int8 with column-wise scaling."""

from __future__ import annotations

import numpy as np


def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float32 vectors to int8, returning quantized vectors and scale factors.

    Symmetric per-dimension (column-wise) quantization. Each dimension is scaled
    so its maximum magnitude maps to 127, the largest representable int8 value —
    i.e. `scaled = v / (max_abs / 127)`, which spans the full [-127, 127] range.

    (Scaling by the max magnitude itself would map everything to [-1, 1] and
    round to just {-1, 0, 1}, wasting 254 of the 256 codes.)
    Returns (int8_vectors, scales) where scales shape is (dimensions,).
    """
    max_abs = np.max(np.abs(vectors), axis=0, keepdims=True).astype(np.float32)  # (1, D)
    # Units-per-code: max magnitude -> 127. Guard all-zero columns.
    scales = np.where(max_abs > 0, max_abs / 127.0, 1e-8).astype(np.float32)

    scaled = vectors / scales  # (N, D) in [-127, 127]
    # Clip in float space *before* casting: casting an out-of-range float to
    # int8 wraps around rather than saturating.
    int8_vectors = np.clip(np.round(scaled), -127, 127).astype(np.int8)

    return int8_vectors, scales.squeeze(0)  # scales shape (D,)


def dequantize_int8(int8_vectors: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize int8 vectors back to float32 using the given scales."""
    scales_reshaped = scales.astype(np.float32).reshape(1, -1)
    return (int8_vectors.astype(np.float32) * scales_reshaped)