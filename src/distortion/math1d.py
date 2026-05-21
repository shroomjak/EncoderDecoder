import numpy as np


# ---------------------------------------------------------------------------
# Геометрия нормировки
# ---------------------------------------------------------------------------

def sensor_center_scale(n_pixels: int) -> tuple[float, float]:
    """x_norm = (x_px - center) / scale"""
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
# Модель дисторсии (division model)
# ---------------------------------------------------------------------------

def distort_division_1d(x_u_norm, k: float) -> np.ndarray:
    """Прямая модель: x_d = x_u / (1 + k·x_u²)"""
    x_u = np.asarray(x_u_norm, dtype=np.float64)
    return x_u / (1.0 + k * x_u**2)


def undistort_division_1d(x_d_norm, k: float) -> np.ndarray:
    """Обратная модель: x_u = 2·x_d / (1 + sqrt(1 - 4·k·x_d²))"""
    x_d = np.asarray(x_d_norm, dtype=np.float64)
    if abs(k) < 1e-15:
        return x_d.copy()
    disc = 1.0 - 4.0 * k * x_d**2
    if np.any(disc < 0):
        raise ValueError("Invalid k: 1 - 4·k·x_d² < 0")
    return 2.0 * x_d / (1.0 + np.sqrt(disc))


# ---------------------------------------------------------------------------
# Построение выходной сетки
# ---------------------------------------------------------------------------

def build_undistorted_grid(
    n_pixels: int,
    k: float,
    output_step_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Равномерная сетка в неискажённом пространстве.

    Returns
    -------
    x_u_norm, x_u_px
    """
    if output_step_px <= 0:
        raise ValueError("output_step_px must be > 0")
    _, scale = sensor_center_scale(n_pixels)
    x_d_left  = pixel_to_norm(0.0, n_pixels)
    x_d_right = pixel_to_norm(float(n_pixels - 1), n_pixels)
    x_u_left  = undistort_division_1d(np.array([x_d_left]),  k)[0]
    x_u_right = undistort_division_1d(np.array([x_d_right]), k)[0]
    step_u   = output_step_px / scale
    x_u_norm = np.arange(x_u_left, x_u_right + 0.5 * step_u, step_u, dtype=np.float64)
    x_u_px   = norm_to_pixel(x_u_norm, n_pixels)
    return x_u_norm, x_u_px


# ---------------------------------------------------------------------------
# Восстановление сигнала
# ---------------------------------------------------------------------------

def restore_signal_1d(
    signal: np.ndarray,
    k: float,
    output_step_px: float = 1.0,
    clip_to_adc: bool = True,
    as_uint16: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Устраняет дисторсию из 1-D сигнала.

    Returns
    -------
    x_u_px    : новые позиции в неискажённой СК
    restored  : восстановленный сигнал
    x_d_px    : соответствующие позиции в исходном сигнале
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < 2:
        raise ValueError("signal length must be >= 2")
    x_u_norm, x_u_px = build_undistorted_grid(n, k, output_step_px)
    x_d_norm = distort_division_1d(x_u_norm, k)
    x_d_px   = norm_to_pixel(x_d_norm, n)
    restored = np.interp(x_d_px, np.arange(n, dtype=np.float64), signal)
    if clip_to_adc:
        restored = np.clip(restored, 0, 4095)
    if as_uint16:
        restored = np.clip(np.rint(restored), 0, 4095).astype(np.uint16)
    return x_u_px, restored, x_d_px


# ---------------------------------------------------------------------------
# Чтение CSV / serial (общий слой I/O)
# ---------------------------------------------------------------------------

def parse_csv_signal_line(line: str) -> tuple[int, np.ndarray]:
    """Строка формата: row_index,v1,v2,...,vN"""
    parts = line.strip().split(",")
    if len(parts) < 2:
        raise ValueError(f"Malformed CSV line: {line!r}")
    row_idx = int(parts[0])
    signal  = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return row_idx, signal


def read_csv_signal_from_open_serial(ser) -> tuple[int, np.ndarray]:
    """Одна строка из уже открытого serial-объекта."""
    line = ser.readline().decode("utf-8", errors="ignore").strip()
    return parse_csv_signal_line(line)


def read_csv_signal_from_port(
    port: str,
    baudrate: int = 115200,
    timeout: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Открывает порт, читает одну строку, закрывает порт."""
    import serial
    with serial.Serial(port, baudrate=baudrate, timeout=timeout) as ser:
        return read_csv_signal_from_open_serial(ser)
