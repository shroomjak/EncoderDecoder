import numpy as np
import matplotlib.pyplot as plt

from src.distortion.math1d import (
    parse_csv_signal_line,
    read_csv_signal_from_open_serial,
    read_csv_signal_from_port,
    restore_signal_1d,
)


# ---------------------------------------------------------------------------
# Восстановление по известному k
# ---------------------------------------------------------------------------

def restore_signal_with_k(
    signal: np.ndarray,
    k: float,
    output_step_px: float = 1.0,
    as_uint16: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x_new, restored, _ = restore_signal_1d(signal, k, output_step_px,
                                            clip_to_adc=True, as_uint16=as_uint16)
    return x_new, restored


# ---------------------------------------------------------------------------
# Удобные обёртки: CSV-строка и serial
# ---------------------------------------------------------------------------

def restore_from_csv_line(
    line: str,
    k: float,
    output_step_px: float = 1.0,
    as_uint16: bool = False,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Вход: 'row_index,v1,...,vN'  →  (row_index, x_new_px, restored)"""
    row_idx, signal = parse_csv_signal_line(line)
    x_new, restored = restore_signal_with_k(signal, k, output_step_px, as_uint16)
    return row_idx, x_new, restored


def restore_from_open_serial(
    ser,
    k: float,
    output_step_px: float = 1.0,
    as_uint16: bool = False,
) -> tuple[int, np.ndarray, np.ndarray]:
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
    """Returns: (row_index, original_signal, x_new_px, restored)"""
    row_idx, signal = read_csv_signal_from_port(port, baudrate, timeout)
    x_new, restored = restore_signal_with_k(signal, k, output_step_px, as_uint16)
    return row_idx, signal, x_new, restored


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def plot_restoration(
    signal: np.ndarray,
    x_new_px: np.ndarray,
    restored: np.ndarray,
    title: str = "Восстановление сигнала",
) -> None:
    signal = np.asarray(signal, dtype=np.float64)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(np.arange(len(signal)), signal,  color="red",   label="Искажённый")
    axs[0].plot(x_new_px,               restored, color="green", alpha=0.8, label="Восстановленный")
    axs[0].set_title(title); axs[0].set_ylabel("ADC")
    axs[0].grid(True); axs[0].legend()

    axs[1].plot(np.arange(len(signal)), signal, color="red", label="Искажённый")
    axs[1].set_xlim(0, len(signal) - 1)
    axs[1].set_title(f"N = {len(signal)},  N' = {len(restored)}")
    axs[1].set_xlabel("px"); axs[1].set_ylabel("ADC")
    axs[1].grid(True); axs[1].legend()

    plt.tight_layout(); plt.show()
