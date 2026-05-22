"""
calibrate_zebra.py — Estimate 1-D radial distortion coefficient k from a
zebra-stripe calibration signal (alternating equal-width bands).

Public API
----------
detect_zebra_edges             — find stripe edges with Blais-Rioux detector
estimate_k_from_zebra_signal   — optimise k by minimising inter-edge spacing variance
estimate_k_from_csv_line       — convenience wrapper for a CSV text line
estimate_k_from_serial         — convenience wrapper reading from a serial port
plot_zebra_calibration         — matplotlib diagnostic plot

ИСПРАВЛЕНИЯ
-----------
BUG 1 (КРИТИЧНЫЙ) — k_bounds по умолчанию (0.0, 0.24) отсекал отрицательные k.
  При подушкообразной дисторсии (k < 0) minimize_scalar с method="bounded"
  молча возвращал k = 0.0 (левую границу) без предупреждения и без ошибки.
  Исправление: изменён default на (-0.24, 0.24).

BUG 2 — Несогласованность реализаций undistort:
  math1d.undistort_division_1d при disc < 0 делает raise ValueError,
  тогда как blais_rioux._undistort_and_resample использует clip(disc, 0) и не падает.
  В _spacing_cost исключение перехватывалось корректно (return inf), но поведение
  на границе было непредсказуемым. Добавлен явный clip(disc, 0) в _undistort_safe(),
  что согласует поведение с blais_rioux и устраняет скачки функции стоимости
  на краях допустимой области.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from src.blais_rioux import (
    BRConfig, detect_edges,
)

from src.distortion.math1d import (
    parse_csv_signal_line,
    read_csv_signal_from_port,
    pixel_to_norm,
    norm_to_pixel, undistort_division_1d,
)

# ---------------------------------------------------------------------------
# k estimation
# ---------------------------------------------------------------------------

def _spacing_cost(k: float, edges_px: np.ndarray, n_pixels: int) -> float:
    """Variance-to-mean² of undistorted inter-edge spacings (minimised at k_opt).

    Вычисляется в пиксельных единицах undistorted-пространства.
    Математически эквивалентно вычислению в нормированных единицах
    (diff(x_u_px) = scale * diff(x_u_norm), scale сокращается),
    но явное использование пикселей делает смысл прозрачным и согласует
    масштаб с остальными метриками (period_px, residual_px).
    """
    x_u_norm = undistort_division_1d(pixel_to_norm(edges_px, n_pixels), k)
    if not np.all(np.isfinite(x_u_norm)):
        return np.inf
    x_u_px = norm_to_pixel(x_u_norm, n_pixels)
    d = np.diff(x_u_px)
    if len(d) < 2:
        return np.inf
    m = np.mean(d)
    return float(np.inf if abs(m) < 1e-15 else np.var(d) / (m ** 2 + 1e-15))


def estimate_k_from_zebra_signal(
    signal: np.ndarray,
    br_config: BRConfig,
    k_bounds: tuple[float, float] = (-0.24, 0.24),   # FIX: было (0.0, 0.24)
    return_debug: bool = False,
):
    """Estimate distortion coefficient k from a single zebra-stripe row.

    Parameters
    ----------
    signal : array
        Raw ADC pixel values.
    br_config : BRConfig
        Blais-Rioux detector configuration (bit_width_px = expected stripe width).
    k_bounds : (k_min, k_max)
        Search bounds for k.
        FIX: нижняя граница изменена с 0.0 на -0.24, чтобы охватывать
        подушкообразную дисторсию (k < 0). При старом значении 0.0
        minimize_scalar молча возвращал k = 0.0 для любого pincushion-объектива.
    return_debug : bool
        If True, return (k, debug_dict).

    Returns
    -------
    k if return_debug is False
    (k, debug_dict) if return_debug is True
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    edges, local_norm, d1, d2, peak_thr = detect_edges(signal, br_config)
    edges_px = np.array([e.position for e in edges], dtype=np.float64)

    if len(edges_px) < 4:
        raise RuntimeError(
            "Too few zebra edges detected. "
            "Adjust BRConfig (bit_width_px, peak_threshold_rel, etc.)."
        )

    res = minimize_scalar(
        _spacing_cost, bounds=k_bounds, method="bounded", args=(edges_px, n)
    )
    k = float(res.x)

    if not return_debug:
        return k

    x_u_norm = _undistort_safe(pixel_to_norm(edges_px, n), k)
    x_u_px   = norm_to_pixel(x_u_norm, n)
    idx      = np.arange(len(x_u_px), dtype=np.float64)
    a_px, b_px = np.polyfit(idx, x_u_px, 1)
    residual_px = x_u_px - (a_px * idx + b_px)

    debug = {
        "edges_px":             edges_px,
        "edges_undistorted_px": x_u_px,
        "residual_px":          residual_px,
        "rms_px":               float(np.sqrt(np.mean(residual_px ** 2))),
        "period_px":            float(a_px),
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
    k_bounds: tuple[float, float] = (-0.24, 0.24),   # FIX: было (0.0, 0.24)
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
    k_bounds: tuple[float, float] = (-0.24, 0.24),   # FIX: было (0.0, 0.24)
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
    signal      = np.asarray(signal, dtype=np.float64)
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
    axs[2].set_title(f"Residuals RMS = {debug['rms_px']:.4f} px")
    axs[2].set_xlabel("Edge index")
    axs[2].set_ylabel("px")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
