"""
math1d.py — 1-D distortion model (division model), signal I/O helpers.

Public API
----------
sensor_center_scale, pixel_to_norm, norm_to_pixel
distort_division_1d, undistort_division_1d
build_undistorted_grid, restore_signal_1d
parse_csv_signal_line, read_csv_signal_from_open_serial, read_csv_signal_from_port
"""
from __future__ import annotations

import time

import serial
import numpy as np

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def sensor_center_scale(n_pixels: int) -> tuple[float, float]:
    """Return (center, scale) so that x_norm = (x_px - center) / scale."""
    if n_pixels < 2:
        raise ValueError("n_pixels must be >= 2")
    center = 0.5 * (n_pixels - 1)
    scale = max(center, 1.0)
    return center, scale


def pixel_to_norm(x_px, n_pixels: int) -> np.ndarray:
    center, scale = sensor_center_scale(n_pixels)
    return (np.asarray(x_px, dtype=np.float64) - center) / scale


def norm_to_pixel(x_norm, n_pixels: int) -> np.ndarray:
    center, scale = sensor_center_scale(n_pixels)
    return np.asarray(x_norm, dtype=np.float64) * scale + center


# ---------------------------------------------------------------------------
# Division distortion model
# ---------------------------------------------------------------------------

def distort_division_1d(x_u_norm, k: float) -> np.ndarray:
    """Forward model: x_d = x_u / (1 + k·x_u²)."""
    x_u = np.asarray(x_u_norm, dtype=np.float64)
    return x_u / (1.0 + k * x_u ** 2)


def undistort_division_1d(x_d_norm, k: float) -> np.ndarray:
    x_d_norm = np.asarray(x_d_norm, dtype=np.float64)
    if abs(k) < 1e-15:
        return x_d_norm.copy()
    disc = np.clip(1.0 - 4.0 * k * x_d_norm ** 2, 0.0, None)
    return 2.0 * x_d_norm / (1.0 + np.sqrt(disc))

# ---------------------------------------------------------------------------
# Undistorted output grid
# ---------------------------------------------------------------------------

def build_undistorted_grid(
    n_pixels: int,
    k: float,
    output_step_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a uniform grid in undistorted space.

    Returns
    -------
    x_u_norm : normalised undistorted coordinates
    x_u_px   : same in pixel units
    """
    if output_step_px <= 0:
        raise ValueError("output_step_px must be > 0")
    _, scale = sensor_center_scale(n_pixels)
    x_u_left  = undistort_division_1d(pixel_to_norm(0.0,             n_pixels), k)[0]
    x_u_right = undistort_division_1d(pixel_to_norm(n_pixels - 1.0, n_pixels), k)[0]
    step_u  = output_step_px / scale
    x_u_norm = np.arange(x_u_left, x_u_right + 0.5 * step_u, step_u, dtype=np.float64)
    return x_u_norm, norm_to_pixel(x_u_norm, n_pixels)


# ---------------------------------------------------------------------------
# Signal restoration
# ---------------------------------------------------------------------------

def restore_signal_1d(
    signal: np.ndarray,
    k: float,
    output_step_px: float = 1.0,
    clip_to_adc: bool = True,
    as_uint16: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Исправление дисторсии 1-D сигнала.

    Алгоритм (геометрически корректный):
    1) есть исходная равномерная сетка x_d_px = 0..n-1 (искажённый сенсор);
    2) считаем x_u_px = undistort(x_d_px) — неравномерные положения тех же точек
       в «идеальном» (недисторсированном) пространстве;
    3) задаём НОВУЮ равномерную сетку x_u_new_px с тем же шагом (output_step_px),
       покрывающую [min(x_u_px), max(x_u_px)];
    4) интерполируем исходный сигнал по x_u_px → значения на x_u_new_px.

    Возвращаем:
        x_u_new_px : равномерная сетка в исправленном пространстве (размер m)
        restored   : значения на этой сетке (размер m)
        x_u_px     : исходная неравномерная сетка позиций для каждого исходного samples
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < 2:
        raise ValueError("signal length must be >= 2")

    # 1. Исходная равномерная сетка (искажённый сенсор)
    x_d_px = np.arange(n, dtype=np.float64)
    x_d_norm = pixel_to_norm(x_d_px, n)

    # 2. Обратное преобразование: координаты в недисторсированном пространстве
    x_u_norm = undistort_division_1d(x_d_norm, k)
    x_u_px = norm_to_pixel(x_u_norm, n)  # НЕравномерная сетка

    # 3. Новая равномерная сетка в исправленном пространстве
    if output_step_px <= 0:
        raise ValueError("output_step_px must be > 0")
    x_min = float(x_u_px.min())
    x_max = float(x_u_px.max())
    n_samples = int(np.floor((x_max - x_min) / output_step_px)) + 1
    x_u_new_px = (x_min + np.arange(n_samples, dtype=np.float64) * output_step_px)

    # 4. Интерполяция сигнала по x_u_px → x_u_new_px
    #    Важно: интерполируем по x_u_px, а не по x_d.
    restored = np.interp(x_u_new_px, x_u_px, signal)

    if clip_to_adc:
        restored = np.clip(restored, 0, 4095)
    if as_uint16:
        restored = np.clip(np.rint(restored), 0, 4095).astype(np.uint16)

    return x_u_new_px, restored, x_u_px


# ---------------------------------------------------------------------------
# CSV / serial I/O
# ---------------------------------------------------------------------------

def parse_csv_signal_line(line: str) -> tuple[int, np.ndarray]:
    """Parse 'row_index,v1,v2,...,vN' → (row_index, signal float32)."""
    parts = line.strip().split(",")
    if len(parts) < 2:
        raise ValueError(f"Malformed CSV line: {line!r}")
    return int(parts[0]), np.array(parts[1:], dtype=np.float32)


def read_csv_signal_from_open_serial(ser) -> tuple[int, np.ndarray]:
    """Read one CSV line from an already-open serial object."""
    line = ser.readline().decode("utf-8", errors="ignore").strip()
    return parse_csv_signal_line(line)


def read_csv_signal_from_port(
    port: str,
    baudrate: int = 115200,
    timeout: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Open serial port, read one line, close."""
    time.sleep(1.0)
    with serial.Serial(port, baudrate=baudrate, timeout=timeout) as ser:
        return read_csv_signal_from_open_serial(ser)
