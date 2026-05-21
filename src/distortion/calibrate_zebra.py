"""
calibrate_zebra.py — Estimate 1-D radial distortion coefficient k from a
zebra-stripe calibration signal (alternating equal-width bands).

Public API
----------
detect_zebra_edges          — find stripe edges with Blais-Rioux detector
estimate_k_from_zebra_signal — optimise k by minimising inter-edge spacing variance
estimate_k_from_csv_line     — convenience wrapper for a CSV text line
estimate_k_from_serial       — convenience wrapper reading from a serial port
plot_zebra_calibration       — matplotlib diagnostic plot
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from src.blais_rioux import (
    BRConfig,
    normalize_global,
    apply_minmax_normalization,
    make_br_kernel,
    correlate_mirror,
    find_zero_crossings_d2,
    filter_by_min_distance,
)
from src.distortion.math1d import (
    parse_csv_signal_line,
    read_csv_signal_from_port,
    pixel_to_norm,
    norm_to_pixel,
    undistort_division_1d,
    sensor_center_scale,
)

# ---------------------------------------------------------------------------
# Edge detection
# ---------------------------------------------------------------------------

def detect_zebra_edges(
    signal: np.ndarray,
    br_config: BRConfig,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, float]:
    """Locate zebra-stripe edges using the Blais-Rioux detector.

    Returns
    -------
    edges, local_norm_signal, d1, d2, peak_threshold
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < 5:
        raise ValueError("Signal is too short (need >= 5 samples)")

    local_norm = apply_minmax_normalization(
        gaussian_filter1d(normalize_global(signal), sigma=br_config.smoothing_sigma, mode="reflect"),
        br_config.minmax_window_px,
    )
    kernel = make_br_kernel(br_config.filter_order)
    d1 = correlate_mirror(local_norm, kernel)
    d2 = correlate_mirror(d1, kernel)

    raw_edges = find_zero_crossings_d2(d2, d1)
    max_amp = max((abs(e.d1_value) for e in raw_edges), default=0.0)
    peak_thr = br_config.peak_threshold_rel * max_amp
    edges = filter_by_min_distance(
        [e for e in raw_edges if abs(e.d1_value) >= peak_thr],
        br_config.bit_width_px * br_config.min_edge_distance_factor,
    )

    roi_start = float(0 if br_config.roi_start is None else br_config.roi_start)
    roi_end   = float(n - 1 if br_config.roi_end is None else br_config.roi_end)
    roi_start, roi_end = sorted([
        float(np.clip(roi_start, 0, n - 1)),
        float(np.clip(roi_end,   0, n - 1)),
    ])
    edges = [e for e in edges if roi_start <= e.position <= roi_end]
    return edges, local_norm, d1, d2, peak_thr


# ---------------------------------------------------------------------------
# k estimation
# ---------------------------------------------------------------------------

def _spacing_cost(k: float, edges_px: np.ndarray, n_pixels: int) -> float:
    """Variance-to-mean² of undistorted inter-edge spacings (minimised at k_opt)."""
    try:
        x_u = undistort_division_1d(pixel_to_norm(edges_px, n_pixels), k)
    except ValueError:
        return np.inf
    if not np.all(np.isfinite(x_u)):
        return np.inf
    d = np.diff(x_u)
    if len(d) < 2:
        return np.inf
    m = np.mean(d)
    return float(np.inf if abs(m) < 1e-15 else np.var(d) / (m ** 2 + 1e-15))


def estimate_k_from_zebra_signal(
    signal: np.ndarray,
    br_config: BRConfig,
    k_bounds: tuple[float, float] = (0.0, 0.24),
    return_debug: bool = False,
):
    """Estimate distortion coefficient k from a single zebra-stripe row.

    Returns
    -------
    k                          if return_debug is False
    (k, debug_dict)            if return_debug is True
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    edges, local_norm, d1, d2, peak_thr = detect_zebra_edges(signal, br_config)
    edges_px = np.array([e.position for e in edges], dtype=np.float64)

    if len(edges_px) < 4:
        raise RuntimeError(
            "Too few zebra edges detected. "
            "Adjust BRConfig (bit_width_px, peak_threshold_rel, etc.)."
        )

    res = minimize_scalar(_spacing_cost, bounds=k_bounds, method="bounded", args=(edges_px, n))
    k = float(res.x)

    if not return_debug:
        return k

    _, scale = sensor_center_scale(n)
    x_d = pixel_to_norm(edges_px, n)
    x_u = undistort_division_1d(x_d, k)
    x_u_px = norm_to_pixel(x_u, n)
    idx = np.arange(len(x_u), dtype=np.float64)
    a, b = np.polyfit(idx, x_u, 1)
    residual_px = (x_u - (a * idx + b)) * scale

    debug = {
        "edges_px":             edges_px,
        "edges_undistorted_px": x_u_px,
        "residual_px":          residual_px,
        "rms_px":               float(np.sqrt(np.mean(residual_px ** 2))),
        "period_px":            float(a * scale),
        "d1":                   d1,
        "d2":                   d2,
        "local_norm_signal":    local_norm,
        "peak_threshold":       peak_thr,
        "objective":            float(res.fun),
        "success":              bool(res.success),
    }
    return k, debug


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def estimate_k_from_csv_line(
    line: str,
    br_config: BRConfig,
    k_bounds: tuple[float, float] = (0.0, 0.24),
    return_debug: bool = False,
):
    row_idx, signal = parse_csv_signal_line(line)
    result = estimate_k_from_zebra_signal(signal, br_config, k_bounds, return_debug)
    if not return_debug:
        return row_idx, result
    k, debug = result
    debug["row_index"] = row_idx
    return row_idx, k, debug


def estimate_k_from_serial(
    port: str,
    br_config: BRConfig,
    baudrate: int = 115200,
    timeout: float = 1.0,
    k_bounds: tuple[float, float] = (0.0, 0.24),
    return_debug: bool = False,
):
    row_idx, signal = read_csv_signal_from_port(port, baudrate, timeout)
    result = estimate_k_from_zebra_signal(signal, br_config, k_bounds, return_debug)
    if not return_debug:
        return row_idx, result
    k, debug = result
    debug["row_index"] = row_idx
    return row_idx, k, debug


# ---------------------------------------------------------------------------
# Diagnostic plot (matplotlib, non-interactive use)
# ---------------------------------------------------------------------------

def plot_zebra_calibration(signal: np.ndarray, debug: dict) -> None:
    """Three-panel diagnostic: raw signal + edges, spacing comparison, residuals."""
    signal = np.asarray(signal, dtype=np.float64)
    edges_px    = debug["edges_px"]
    edges_undist = debug["edges_undistorted_px"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(signal, color="black", lw=1.0, label="ADC signal")
    for x in edges_px:
        axs[0].axvline(x, color="red", alpha=0.2)
    axs[0].set_title("Zebra signal and detected edges")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(np.diff(edges_px),    "o-", color="red",   label="Before correction")
    axs[1].plot(np.diff(edges_undist), "x-", color="green", label="After correction")
    axs[1].set_title("Inter-edge spacings")
    axs[1].set_ylabel("px")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(debug["residual_px"], "o-", color="purple")
    axs[2].axhline(0.0, color="gray", lw=1.0)
    axs[2].set_title(f"Residuals  RMS = {debug['rms_px']:.4f} px")
    axs[2].set_xlabel("Edge index")
    axs[2].set_ylabel("px")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
