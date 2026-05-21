"""
restore_signal.py — Distortion correction with OpenCV real-time visualisation.

Two rendering modes
-------------------
* **CSV mode**   — read rows from a text file, show frame-by-frame.
* **Serial mode** — stream live frames from a serial port (CSV lines).

Window layout (identical to demo_opencv_br.py)
-----------------------------------------------
┌─────────────────────────────────────────────────┐
│ HEADER  — row index, k, pixel counts, range     │
├─────────────────────────────────────────────────┤
│ ORIGINAL  — pseudo-2D grayscale strip (raw)     │
├─────────────────────────────────────────────────┤
│ RESTORED  — pseudo-2D grayscale strip (fixed)   │
└─────────────────────────────────────────────────┘

Public API (library use)
------------------------
restore_signal_with_k
restore_from_csv_line, restore_from_open_serial, restore_from_serial
draw_strip_2d
show_restoration_opencv        — single-frame window
run_restoration_from_csv       — CSV file loop
run_restoration_from_serial    — serial streaming loop
plot_restoration               — matplotlib static plot
"""
from __future__ import annotations

import sys
from typing import Optional

import cv2
import numpy as np

from src.distortion.math1d import (
    parse_csv_signal_line,
    read_csv_signal_from_open_serial,
    read_csv_signal_from_port,
    restore_signal_1d,
)


# ===========================================================================
# AutoRange — EMA-smoothed display range (shared with demo_opencv_br pattern)
# ===========================================================================

class AutoRange:
    """Exponentially smoothed [lo, hi] range for auto-scaling display."""

    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self._lo: Optional[float] = None
        self._hi: Optional[float] = None

    def reset(self) -> None:
        self._lo = self._hi = None

    def update(self, values: np.ndarray) -> tuple[float, float]:
        lo = float(np.percentile(values, 1))
        hi = float(np.percentile(values, 99))
        if self._lo is None:
            self._lo, self._hi = lo, hi
        else:
            a = self.alpha
            self._lo = (1 - a) * self._lo + a * lo
            self._hi = (1 - a) * self._hi + a * hi
        if self._hi - self._lo < 1e-6:
            self._hi = self._lo + 1.0
        return self._lo, self._hi


# ===========================================================================
# Drawing helpers
# ===========================================================================

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_COLOR_BG   = (245, 245, 245)
_COLOR_TEXT = (30,  30,  30 )
_COLOR_MUT  = (120, 120, 120)
_COLOR_OK   = (40,  160, 40 )
_COLOR_EDGE = (100, 100, 100)


def _put(img: np.ndarray, text: str, x: int, y: int,
         color=_COLOR_TEXT, scale: float = 0.45, thickness: int = 1) -> None:
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def normalize_pixels(
    pixels: np.ndarray,
    lo: float,
    hi: float,
    invert: bool = False,
) -> np.ndarray:
    """Normalise pixel values to uint8 [0, 255]."""
    scaled = np.clip((pixels - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray = (scaled * 255).astype(np.uint8)
    return 255 - gray if invert else gray


def draw_strip_2d(
    pixels: np.ndarray,
    lo: float,
    hi: float,
    width: int,
    height: int,
    invert: bool = False,
) -> np.ndarray:
    """Pseudo-2D grayscale strip — horizontal CCD signal visualisation.

    Equivalent to the upper panel in demo_opencv_br and ax_2d in visualizer.py.
    The 1-D signal is repeated vertically to *height* rows; if the resulting
    width differs from *width* it is resampled with INTER_NEAREST.
    """
    gray  = normalize_pixels(pixels, lo, hi, invert=invert)
    strip = np.repeat(gray[np.newaxis, :], height, axis=0)
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    if strip_bgr.shape[1] != width:
        strip_bgr = cv2.resize(strip_bgr, (width, height),
                               interpolation=cv2.INTER_NEAREST)
    return strip_bgr


def _draw_header(
    row_idx: int,
    k: float,
    n_orig: int,
    n_rest: int,
    lo: float,
    hi: float,
    width: int,
    height: int,
    source_label: str = "",
) -> np.ndarray:
    hdr = np.full((height, width, 3), 245, dtype=np.uint8)
    lines = [
        (f"source={source_label}  row={row_idx}  "
         f"range=[{lo:.1f}, {hi:.1f}]",       _COLOR_MUT,  0.44),
        (f"k={k:.6f}  N_orig={n_orig}  N_restored={n_rest}", _COLOR_TEXT, 0.44),
    ]
    y = 18
    for text, color, scale in lines:
        _put(hdr, text, 8, y, color=color, scale=scale)
        y += 18
    cv2.line(hdr, (0, height - 1), (width - 1, height - 1), _COLOR_EDGE, 1)
    return hdr


def _compose_frame(
    row_idx: int,
    original: np.ndarray,
    restored: np.ndarray,
    k: float,
    lo: float,
    hi: float,
    window_width: int,
    header_h: int,
    strip_h: int,
    source_label: str = "",
    invert: bool = False,
) -> np.ndarray:
    hdr  = _draw_header(row_idx, k, len(original), len(restored),
                        lo, hi, window_width, header_h, source_label)
    s_orig = draw_strip_2d(original, lo, hi, window_width, strip_h, invert)
    s_rest = draw_strip_2d(restored, lo, hi, window_width, strip_h, invert)

    # Label each strip
    _put(s_orig, "ORIGINAL", 4, 14, color=_COLOR_MUT, scale=0.4)
    _put(s_rest, "RESTORED", 4, 14, color=_COLOR_OK,  scale=0.4)

    # Divider line between strips
    div = np.full((2, window_width, 3), 180, dtype=np.uint8)
    return np.vstack([hdr, s_orig, div, s_rest])


# ===========================================================================
# Core restoration helpers (library API)
# ===========================================================================

def restore_signal_with_k(
    signal: np.ndarray,
    k: float,
    output_step_px: float = 1.0,
    as_uint16: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply division-model undistortion.

    Returns
    -------
    x_new_px : output pixel positions in undistorted frame
    restored : corrected signal values
    """
    x_new, restored, _ = restore_signal_1d(
        signal, k, output_step_px, clip_to_adc=True, as_uint16=as_uint16
    )
    return x_new, restored


def restore_from_csv_line(
    line: str,
    k: float,
    output_step_px: float = 1.0,
    as_uint16: bool = False,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Parse CSV line and restore signal.

    Returns (row_index, x_new_px, restored).
    """
    row_idx, signal = parse_csv_signal_line(line)
    x_new, restored = restore_signal_with_k(signal, k, output_step_px, as_uint16)
    return row_idx, x_new, restored


def restore_from_open_serial(
    ser,
    k: float,
    output_step_px: float = 1.0,
    as_uint16: bool = False,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Read one CSV line from open serial and restore signal."""
    row_idx, signal = read_csv_signal_from_open_serial(ser)
    x_new, restored = restore_signal_with_k(signal, k, output_step_px, as_uint16)
    return row_idx, x_new, restored


def restore_from_serial(
    port: str,
    k: float,
    baudrate: int = 115200,
    timeout: float = 1.0,
    output_step_px: float = 1.0,
    as_uint16: bool = False,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Open serial port, read one row, restore and return.

    Returns (row_index, original_signal, x_new_px, restored).
    """
    row_idx, signal = read_csv_signal_from_port(port, baudrate, timeout)
    x_new, restored = restore_signal_with_k(signal, k, output_step_px, as_uint16)
    return row_idx, signal, x_new, restored


# ===========================================================================
# OpenCV visualisation — single frame
# ===========================================================================

def show_restoration_opencv(
    signal: np.ndarray,
    restored: np.ndarray,
    k: float,
    row_idx: int = 0,
    source_label: str = "csv",
    window_width: int = 900,
    header_h: int = 60,
    strip_h: int = 48,
    invert: bool = False,
    window_name: str = "Signal Restoration",
    wait_ms: int = 0,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Display a single-frame restoration result in an OpenCV window.

    Returns the composed frame array (BGR).
    """
    ranger = AutoRange()
    lo, hi = ranger.update(np.concatenate([signal, restored]))
    frame = _compose_frame(row_idx, signal, restored, k, lo, hi,
                           window_width, header_h, strip_h, source_label, invert)
    if save_path:
        cv2.imwrite(save_path, frame)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    cv2.waitKey(wait_ms)
    return frame


# ===========================================================================
# OpenCV visualisation — CSV file loop
# ===========================================================================

def run_restoration_from_csv(
    csv_path: str,
    k: float,
    output_step_px: float = 1.0,
    window_width: int = 900,
    header_h: int = 60,
    strip_h: int = 48,
    invert: bool = False,
    frame_delay_ms: int = 30,
    window_name: str = "Signal Restoration — CSV",
) -> None:
    """Iterate over CSV rows and display each frame in an OpenCV window.

    Press **q** or **Esc** to quit.
    """
    ranger = AutoRange()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with open(csv_path, "r") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row_idx, signal = parse_csv_signal_line(raw_line)
                _, restored = restore_signal_with_k(signal, k, output_step_px)
            except Exception as exc:
                print(f"[WARN] Skipping line: {exc}", file=sys.stderr)
                continue

            lo, hi = ranger.update(signal)
            frame = _compose_frame(row_idx, signal, restored, k, lo, hi,
                                   window_width, header_h, strip_h,
                                   source_label="csv", invert=invert)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break

    cv2.destroyAllWindows()


# ===========================================================================
# OpenCV visualisation — serial streaming loop (real-time)
# ===========================================================================

def run_restoration_from_serial(
    port: str,
    k: float,
    baudrate: int = 115200,
    timeout: float = 1.0,
    output_step_px: float = 1.0,
    window_width: int = 900,
    header_h: int = 60,
    strip_h: int = 48,
    invert: bool = False,
    window_name: str = "Signal Restoration — Serial",
    save_first_frame: Optional[str] = None,
) -> None:
    """Stream live frames from a serial port and display in an OpenCV window.

    Each line must be CSV-encoded as ``row_index,v1,v2,...,vN``.
    Press **q** or **Esc** to stop.
    """
    import serial as _serial

    ranger = AutoRange()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        with _serial.Serial(port, baudrate=baudrate, timeout=timeout) as ser:
            while True:
                try:
                    row_idx, signal = read_csv_signal_from_open_serial(ser)
                    _, restored = restore_signal_with_k(signal, k, output_step_px)
                except Exception as exc:
                    print(f"[WARN] {exc}", file=sys.stderr)
                    continue

                lo, hi = ranger.update(signal)
                frame = _compose_frame(row_idx, signal, restored, k, lo, hi,
                                       window_width, header_h, strip_h,
                                       source_label=port, invert=invert)

                if save_first_frame:
                    cv2.imwrite(save_first_frame, frame)
                    save_first_frame = None          # save once

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()


# ===========================================================================
# Matplotlib static plot (offline / notebook use)
# ===========================================================================

def plot_restoration(
    signal: np.ndarray,
    x_new_px: np.ndarray,
    restored: np.ndarray,
    title: str = "Signal restoration",
) -> None:
    """Two-panel matplotlib figure comparing original and restored signals."""
    import matplotlib.pyplot as plt

    signal = np.asarray(signal, dtype=np.float64)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    axs[0].plot(np.arange(len(signal)), signal,   color="red",   label="Distorted")
    axs[0].plot(x_new_px,               restored,  color="green", alpha=0.8, label="Restored")
    axs[0].set_title(title)
    axs[0].set_ylabel("ADC")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(np.arange(len(signal)), signal, color="red", label="Distorted (original axis)")
    axs[1].set_xlim(0, len(signal) - 1)
    axs[1].set_title(f"N_orig={len(signal)}  N_restored={len(restored)}")
    axs[1].set_xlabel("px")
    axs[1].set_ylabel("ADC")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
