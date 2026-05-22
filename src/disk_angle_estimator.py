"""
disk_angle_estimator.py — расчёт абсолютного угла диска по кодовой дорожке
абсолютного энкодера с де-Брейновской подпоследовательностью.

Алгоритм (estimate_disk_angle_from_result)
------------------------------------------
Входные данные: BitRecoveryResult det, code_angle_map, параметры диска.

Шаг 1. Масштаб пиксель/угол вычисляется из самого det:
    mean_period_px = среднее measured_period по всем BitSegment с весом n_bits
    delta_phi      = angle_period_deg / total_code_bits
    angle_per_px   = delta_phi / mean_period_px   [°/px]

Шаг 2. Из det.recovered_bits извлекаются валидные биты в порядке возрастания
position. Для каждого бита i:
    x_i = position_i + 0.5 * measured_period_i   (центр бита в пикселях)
    w_i = measured_period_i                      (вес = ширина бита)

Шаг 3. По всем скользящим окнам длины m = codeword_length:
    - собирается наблюдаемое кодовое слово;
    - ищется в code_angle_map (при reverse_direction карта перестроена
      с реверсированной последовательностью);
    - по битам окна считается взвешенный центр окна на сенсоре;
    - получается оценка абсолютного угла диска:
          theta = phi_win_center - signed_angle_per_px * (x_win_center - sensor_center_px)

    signed_angle_per_px = +angle_per_px  при reverse_direction=False
    signed_angle_per_px = -angle_per_px  при reverse_direction=True

Шаг 4. Финал:
    mean_angle_deg = circular_mean(theta_s)
    std_angle_deg  = circular_std(theta_s)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.blais_rioux import BitRecoveryResult


REVERSE_AXIS_SIGN = True

FULL_DISK_CODE_SEQUENCE = (
    "100001011100001100100001101100001110100001111100010001100010010100"
    "010011100010100100010101100010110100010111100011001100011010100011"
    "011100011100100011101100011110100011111100100100101100100110100100"
    "111100101001100101010100101011100101101100101110100101111100110011"
    "100110101100110110100110111100111010100111011100111101100111110100"
    "111111101010101101010111101011011101011101101011111101101101111101"
    "1101111011111111"
)
TOTAL_CODE_BITS_ON_DISK = len(FULL_DISK_CODE_SEQUENCE)
CODEWORD_LENGTH_BITS = 9
ANGLE_PERIOD_DEG = 360.0


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class AngleWindowSample:
    """Оценка угла по одному совпавшему кодовому окну."""
    window_start: int
    codeword: str
    window_center_px: float
    window_angle_deg: float
    estimated_angle_deg: float
    start_bit_index: int = 0


@dataclass
class AngleEstimationResult:
    """Результат оценки абсолютного угла диска."""
    codeword_length: int
    visible_bits: int
    total_windows: int
    matched_windows: int

    mean_bit_period_px: Optional[float] = None
    angle_per_px_deg: Optional[float] = None
    reverse_direction: bool = False

    samples: List[AngleWindowSample] = field(default_factory=list)
    mean_angle_deg: Optional[float] = None
    std_angle_deg: Optional[float] = None


# ---------------------------------------------------------------------------
# Вспомогательные функции — углы
# ---------------------------------------------------------------------------

def wrap_angle_deg(angle_deg: float, angle_period_deg: float = 360.0) -> float:
    """Нормализация угла в [0, angle_period_deg)."""
    if angle_period_deg <= 0:
        raise ValueError("angle_period_deg must be > 0")
    return float(angle_deg % angle_period_deg)


def angle_in_range_deg(
    angle_deg: float,
    lo_deg: float,
    hi_deg: float,
    angle_period_deg: float = 360.0,
) -> bool:
    """Попадает ли угол в [lo_deg, hi_deg) на окружности."""
    a = wrap_angle_deg(angle_deg, angle_period_deg)
    lo = wrap_angle_deg(lo_deg, angle_period_deg)
    hi = wrap_angle_deg(hi_deg, angle_period_deg)
    if lo <= hi:
        return lo <= a < hi
    return a >= lo or a < hi


def circular_mean_deg(
    angles_deg: np.ndarray,
    weights: Optional[np.ndarray] = None,
    angle_period_deg: float = 360.0,
) -> Optional[float]:
    """Circular mean для углов в градусах."""
    if angles_deg is None or len(angles_deg) == 0:
        return None
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(angles_deg)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != len(angles_deg):
            raise ValueError("weights and angles_deg must have the same length")
    wsum = np.sum(weights)
    if wsum <= 1e-12:
        return None
    phase = angles_deg * (2.0 * np.pi / angle_period_deg)
    s = np.sum(weights * np.sin(phase))
    c = np.sum(weights * np.cos(phase))
    if abs(s) <= 1e-12 and abs(c) <= 1e-12:
        return None
    mean_phase = np.arctan2(s, c)
    if mean_phase < 0:
        mean_phase += 2.0 * np.pi
    return wrap_angle_deg(mean_phase * angle_period_deg / (2.0 * np.pi), angle_period_deg)


def circular_std_deg(
    angles_deg: np.ndarray,
    weights: Optional[np.ndarray] = None,
    angle_period_deg: float = 360.0,
) -> Optional[float]:
    """Circular std для углов в градусах."""
    if angles_deg is None or len(angles_deg) == 0:
        return None
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(angles_deg)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != len(angles_deg):
            raise ValueError("weights and angles_deg must have the same length")
    wsum = np.sum(weights)
    if wsum <= 1e-12:
        return None
    phase = angles_deg * (2.0 * np.pi / angle_period_deg)
    s = np.sum(weights * np.sin(phase))
    c = np.sum(weights * np.cos(phase))
    r = float(np.clip(np.hypot(s, c) / wsum, 0.0, 1.0 - 1e-15))
    if r <= 1e-12:
        return None
    return float(np.sqrt(-2.0 * np.log(r)) * angle_period_deg / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Кодовая карта
# ---------------------------------------------------------------------------

def _validate_binary_code_sequence(code_sequence: str, total_code_bits: int) -> str:
    if total_code_bits <= 0:
        raise ValueError("total_code_bits must be > 0")
    if code_sequence is None:
        raise ValueError("code_sequence is None")
    seq = code_sequence.strip()
    if len(seq) != total_code_bits:
        raise ValueError(
            f"len(code_sequence)={len(seq)} != total_code_bits={total_code_bits}"
        )
    if any(ch not in "01" for ch in seq):
        raise ValueError("code_sequence must contain only '0' and '1'")
    return seq


def build_code_angle_map(
    code_sequence: str,
    total_code_bits: int,
    codeword_length: int,
    angle_period_deg: float = 360.0,
) -> Dict[str, dict]:
    """
    Строит карту codeword -> {start_bit_index, angle_lo_deg, angle_center_deg, angle_hi_deg}.

    Углы хранятся в линейном виде (без wrap), чтобы арифметика
    вблизи 0°/360° не вносила разрывов.
    """
    seq = _validate_binary_code_sequence(code_sequence, total_code_bits)
    if codeword_length <= 0:
        raise ValueError("codeword_length must be > 0")
    if codeword_length > total_code_bits:
        raise ValueError(
            f"codeword_length={codeword_length} > total_code_bits={total_code_bits}"
        )
    if angle_period_deg <= 0:
        raise ValueError("angle_period_deg must be > 0")

    delta = angle_period_deg / total_code_bits
    code_map: Dict[str, dict] = {}
    duplicates: Dict[str, List[int]] = {}

    for start_idx in range(total_code_bits):
        codeword = "".join(
            seq[(start_idx + k) % total_code_bits]
            for k in range(codeword_length)
        )
        entry = {
            "start_bit_index": start_idx,
            "angle_lo_deg": start_idx * delta,
            "angle_center_deg": (start_idx + codeword_length / 2.0) * delta,
            "angle_hi_deg": (start_idx + codeword_length) * delta,
        }
        if codeword in code_map:
            duplicates.setdefault(
                codeword, [code_map[codeword]["start_bit_index"]]
            ).append(start_idx)
        else:
            code_map[codeword] = entry

    if duplicates:
        details = "; ".join(
            f"{cw}: {starts}" for cw, starts in list(duplicates.items())[:8]
        )
        if len(duplicates) > 8:
            details += "; ..."
        raise ValueError(
            f"Кодовая карта неоднозначна — дублирующиеся слова: {details}"
        )
    return code_map


def print_code_angle_table(
    code_angle_map: Dict[str, dict],
    filter_angle_lo: Optional[float] = None,
    filter_angle_hi: Optional[float] = None,
    angle_period_deg: float = 360.0,
    show_start_bit_index: bool = False,
) -> None:
    """Печатает таблицу кодовой карты, сортировка по углу центра."""
    if not code_angle_map:
        print("(code_angle_map is empty)")
        return

    w = len(next(iter(code_angle_map.keys())))

    if show_start_bit_index:
        header = (
            f"  {'Угол_нижн, °':>13} | {'Угол_центр, °':>14} | {'Угол_верхн, °':>14} | "
            f"{'StartIdx':>8} | {'Код':>{w}}"
        )
    else:
        header = (
            f"  {'Угол_нижн, °':>13} | {'Угол_центр, °':>14} | {'Угол_верхн, °':>14} | "
            f"{'Код':>{w}}"
        )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    sorted_items = sorted(
        code_angle_map.items(),
        key=lambda kv: kv[1]["angle_center_deg"] % angle_period_deg,
    )

    for codeword, entry in sorted_items:
        lo_w = wrap_angle_deg(entry["angle_lo_deg"], angle_period_deg)
        c_w = wrap_angle_deg(entry["angle_center_deg"], angle_period_deg)
        hi_w = wrap_angle_deg(entry["angle_hi_deg"], angle_period_deg)

        if filter_angle_lo is not None and filter_angle_hi is not None:
            if not angle_in_range_deg(c_w, filter_angle_lo, filter_angle_hi, angle_period_deg):
                continue

        if show_start_bit_index:
            print(
                f"  {lo_w:>13.4f} | {c_w:>14.4f} | {hi_w:>14.4f} | "
                f"{entry['start_bit_index']:>8d} | {codeword:>{w}}"
            )
        else:
            print(
                f"  {lo_w:>13.4f} | {c_w:>14.4f} | {hi_w:>14.4f} | {codeword:>{w}}"
            )
    print(sep)


# ---------------------------------------------------------------------------
# Основная функция оценки угла
# ---------------------------------------------------------------------------

def estimate_disk_angle_from_result(
    detection_result: "BitRecoveryResult",
    code_sequence: str,
    total_code_bits: int,
    codeword_length: int,
    sensor_center_px: float,
    angle_period_deg: float = 360.0,
) -> AngleEstimationResult:
    """
    Оценка абсолютного угла диска по видимому участку кода.

    При reverse_direction=True карта перестраивается с реверсированной
    кодовой последовательностью. Знак angle_per_px меняется.

    Parameters
    ----------
    detection_result : BitRecoveryResult
    code_sequence : str
        Полная кодовая последовательность на диске.
    total_code_bits : int
        Число бит на полном диске N.
    codeword_length : int
        Длина кодового слова m.
    sensor_center_px : float
        Опорный центр сенсора в пикселях.
    angle_period_deg : float
        Полный угловой период.
    """
    if codeword_length <= 0:
        raise ValueError("codeword_length must be > 0")
    if total_code_bits <= 0:
        raise ValueError("total_code_bits must be > 0")
    if angle_period_deg <= 0:
        raise ValueError("angle_period_deg must be > 0")

    # --- Шаг 1. Масштаб из det ---
    total_px = 0.0
    total_n = 0
    for seg in detection_result.bit_segments:
        n = int(seg.n_bits)
        p = float(seg.measured_period)
        if n <= 0 or p <= 1e-12:
            continue
        total_px += p * n
        total_n += n

    mean_period_px = (total_px / total_n) if total_n > 0 else None

    result = AngleEstimationResult(
        codeword_length=codeword_length,
        visible_bits=0,
        total_windows=0,
        matched_windows=0,
        mean_bit_period_px=mean_period_px,
        reverse_direction=REVERSE_AXIS_SIGN,
    )

    if mean_period_px is None or mean_period_px <= 1e-12:
        return result

    delta_phi = angle_period_deg / total_code_bits
    angle_per_px_abs = delta_phi / mean_period_px
    result.angle_per_px_deg = angle_per_px_abs

    # --- Построение карты (реверс последовательности при необходимости) ---
    effective_sequence = code_sequence[::-1] if REVERSE_AXIS_SIGN else code_sequence
    code_map = build_code_angle_map(
        effective_sequence, total_code_bits, codeword_length, angle_period_deg
    )

    # --- Шаг 2. Извлечение валидных бит ---
    bit_values: List[int] = []
    bit_centers_px: List[float] = []
    bit_widths_px: List[float] = []

    for rb in sorted(detection_result.recovered_bits, key=lambda b: b.position):
        if not (0 <= rb.segment_idx < len(detection_result.bit_segments)):
            continue
        seg = detection_result.bit_segments[rb.segment_idx]
        period_px = float(seg.measured_period)
        if period_px <= 1e-12:
            continue
        bit_start = float(rb.position)
        bit_end = bit_start + period_px
        eps = max(1e-9, 1e-6 * period_px)
        if bit_start < float(seg.start_pos) - eps:
            continue
        if bit_end > float(seg.end_pos) + eps:
            continue
        bit_values.append(int(rb.value))
        bit_centers_px.append(bit_start + 0.5 * period_px)
        bit_widths_px.append(period_px)

    n_bits = len(bit_values)
    total_windows = max(0, n_bits - codeword_length + 1)
    result.visible_bits = n_bits
    result.total_windows = total_windows

    if n_bits == 0 or total_windows == 0:
        return result

    centers = np.asarray(bit_centers_px, dtype=np.float64)
    widths = np.asarray(bit_widths_px, dtype=np.float64)

    # --- Шаг 3. Скользящие окна ---
    samples: List[AngleWindowSample] = []
    angle_values_deg: List[float] = []

    for s in range(total_windows):
        codeword = "".join(str(bit_values[s + k]) for k in range(codeword_length))

        entry = code_map.get(codeword)
        if entry is None:
            continue

        w_win = widths[s: s + codeword_length]
        c_win = centers[s: s + codeword_length]
        wsum = float(np.sum(w_win))
        if wsum <= 1e-12:
            continue

        x_win_center = float(np.sum(w_win * c_win) / wsum)
        phi_win_center = entry["angle_center_deg"]

        theta = phi_win_center - angle_per_px_abs * (x_win_center - sensor_center_px)
        theta_wrapped = wrap_angle_deg(theta, angle_period_deg)

        angle_values_deg.append(theta_wrapped)
        samples.append(AngleWindowSample(
            window_start=s,
            codeword=codeword,
            window_center_px=x_win_center,
            window_angle_deg=wrap_angle_deg(phi_win_center, angle_period_deg),
            estimated_angle_deg=theta_wrapped,
            start_bit_index=entry["start_bit_index"],
        ))

    result.samples = samples
    result.matched_windows = len(samples)

    if not angle_values_deg:
        return result

    # --- Шаг 4. Circular mean / std ---
    angles_arr = np.asarray(angle_values_deg, dtype=np.float64)
    result.mean_angle_deg = circular_mean_deg(angles_arr, angle_period_deg=angle_period_deg)
    result.std_angle_deg = circular_std_deg(angles_arr, angle_period_deg=angle_period_deg)

    print("angles arr", angles_arr)
    return result


if __name__ == "__main__":
    code_map = build_code_angle_map(
        FULL_DISK_CODE_SEQUENCE,
        TOTAL_CODE_BITS_ON_DISK,
        CODEWORD_LENGTH_BITS,
        ANGLE_PERIOD_DEG,
    )

    print("Фильтр по углу центра окна: 90°..95°")
    print_code_angle_table(
        code_map, 90.0, 95.0, ANGLE_PERIOD_DEG, True
    )