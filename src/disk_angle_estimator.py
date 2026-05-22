"""
disk_angle_estimator.py — расчёт абсолютного угла диска по кодовой дорожке
абсолютного энкодера с де-Брейновской подпоследовательностью.

Новая схема оценки
------------------
1. Из BitRecoveryResult извлекаются восстановленные биты; для каждого
   вычисляются:
     - значение бита,
     - центр бита в пикселях:
           x_i = position + 0.5 * measured_period
     - вес бита по ширине:
           w_i = measured_period

2. По всем валидным recovered_bits вычисляется единый центр всего видимого
   участка кода на сенсоре:
       x_vis = sum(w_i * x_i) / sum(w_i)

   Это агрегированная оценка положения видимого кода по всем пикселам.

3. По всем скользящим окнам длины m выполняется поиск в code_angle_map.
   Для каждого совпавшего окна с локальным стартом s:
       phi_win = angle_center_deg(codeword)
           угол центра найденного кодового окна на диске

       phi_vis = phi_win + (n_bits/2 - s - m/2) * delta_phi
           угол центра ВСЕГО видимого участка на диске

       theta_s = phi_vis - angle_per_px_deg * (x_vis - sensor_center_px)
           оценка абсолютного угла диска

4. Финальный результат:
       mean_angle_deg = circmean(theta_s)
       std_angle_deg  = circstd(theta_s)

Замечание
---------
В этом алгоритме геометрия изображения оценивается один раз по всему видимому
фрагменту, а разные matched windows используются только для абсолютной
дисковой привязки и оценки согласованности.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.blais_rioux import BitRecoveryResult


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

@dataclass(frozen=True)
class CodeAngleEntry:
    """
    Угловая информация о кодовом слове.

    Основной смысл записи — угловое положение кодового окна.
    start_bit_index оставлен только как служебная диагностическая информация.
    """
    codeword: str
    start_bit_index: int
    angle_lo_deg: float
    angle_center_deg: float
    angle_hi_deg: float


@dataclass
class AngleWindowSample:
    """Оценка угла по одному совпавшему кодовому окну."""
    window_start: int                  # локальный индекс начала окна в visible bits
    codeword: str                      # найденное кодовое слово
    matched_window_angle_deg: float    # угол центра найденного окна на диске
    visible_code_angle_deg: float      # выведенный угол центра всего видимого участка
    visible_code_center_px: float      # агрегированный центр видимого участка на сенсоре
    estimated_angle_deg: float         # оценка абсолютного угла диска
    start_bit_index: Optional[int] = None


@dataclass
class AngleEstimationResult:
    """Результат оценки абсолютного угла диска."""
    codeword_length: int
    visible_bits: int
    total_windows: int
    matched_windows: int

    visible_code_center_px: Optional[float] = None
    visible_code_span_px: Optional[float] = None

    samples: List[AngleWindowSample] = field(default_factory=list)
    mean_angle_deg: Optional[float] = None
    std_angle_deg: Optional[float] = None


# ---------------------------------------------------------------------------
# Вспомогательные функции
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
    """
    Проверяет, попадает ли угол в полуинтервал [lo_deg, hi_deg) на окружности.
    Поддерживает диапазоны с переходом через 0°.
    """
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
    """Circular mean (Mardia & Jupp) для углов в градусах."""
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
    """Circular std (Mardia & Jupp) для углов в градусах."""
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


def _validate_binary_code_sequence(code_sequence: str, total_code_bits: int) -> str:
    """Проверка и нормализация кодовой последовательности."""
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


def _extract_valid_recovered_bits_geometry(
    detection_result: "BitRecoveryResult",
) -> tuple[List[int], List[float], List[float]]:
    """
    Извлекает валидные recovered bits в порядке возрастания position.

    Returns
    -------
    bit_values : List[int]
        Значения бит.
    bit_centers_px : List[float]
        Центры битов в пикселях:
            position + 0.5 * measured_period
    bit_widths_px : List[float]
        Ширины битов в пикселях:
            measured_period
    """
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

        # Бит должен лежать внутри своего сегмента
        if bit_start < float(seg.start_pos) - eps:
            continue
        if bit_end > float(seg.end_pos) + eps:
            continue

        bit_values.append(int(rb.value))
        bit_centers_px.append(bit_start + 0.5 * period_px)
        bit_widths_px.append(period_px)

    return bit_values, bit_centers_px, bit_widths_px


def _compute_visible_code_center_px(
    bit_centers_px: List[float],
    bit_widths_px: List[float],
) -> tuple[Optional[float], Optional[float]]:
    """
    Вычисляет агрегированный центр видимого кода по всем пикселам.

    Используется период-взвешенный центр:
        x_vis = sum(w_i * x_i) / sum(w_i)

    где:
        x_i = центр i-го бита,
        w_i = measured_period i-го бита.

    Для непрерывной цепочки бит это соответствует геометрическому центру
    всего видимого участка.

    Returns
    -------
    (visible_code_center_px, visible_code_span_px)
    """
    if not bit_centers_px or not bit_widths_px:
        return None, None

    centers = np.asarray(bit_centers_px, dtype=np.float64)
    widths = np.asarray(bit_widths_px, dtype=np.float64)

    wsum = float(np.sum(widths))
    if wsum <= 1e-12:
        return None, None

    x_vis = float(np.sum(widths * centers) / wsum)
    return x_vis, wsum


# ---------------------------------------------------------------------------
# Построение кодовой карты
# ---------------------------------------------------------------------------

def build_code_angle_map(
    code_sequence: str,
    total_code_bits: int,
    codeword_length: int,
    *,
    angle_period_deg: float = 360.0,
) -> Dict[str, CodeAngleEntry]:
    """
    Строит карту:
        codeword -> CodeAngleEntry

    Карта является угловой: основное значение записи — положение кодового окна
    на диске в угловых единицах.

    Parameters
    ----------
    code_sequence : str
        Полная циклическая последовательность длины N.
    total_code_bits : int
        N.
    codeword_length : int
        m.
    angle_period_deg : float
        Полный угловой период.

    Returns
    -------
    Dict[str, CodeAngleEntry]
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

    code_map: Dict[str, CodeAngleEntry] = {}
    duplicates: Dict[str, List[int]] = {}

    for start_idx in range(total_code_bits):
        codeword = "".join(
            seq[(start_idx + k) % total_code_bits]
            for k in range(codeword_length)
        )

        angle_lo = wrap_angle_deg(start_idx * delta, angle_period_deg)
        angle_center = wrap_angle_deg(
            (start_idx + codeword_length / 2.0) * delta,
            angle_period_deg,
        )
        angle_hi = wrap_angle_deg(
            (start_idx + codeword_length) * delta,
            angle_period_deg,
        )

        entry = CodeAngleEntry(
            codeword=codeword,
            start_bit_index=start_idx,
            angle_lo_deg=angle_lo,
            angle_center_deg=angle_center,
            angle_hi_deg=angle_hi,
        )

        if codeword in code_map:
            duplicates.setdefault(
                codeword, [code_map[codeword].start_bit_index]
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


# ---------------------------------------------------------------------------
# Просмотр кодовой карты
# ---------------------------------------------------------------------------

def lookup_angle_by_code(
    codeword: str,
    code_angle_map: Dict[str, CodeAngleEntry],
) -> Optional[CodeAngleEntry]:
    """Возвращает угловую запись для кодового слова."""
    return code_angle_map.get(codeword)


def print_code_angle_table(
    code_angle_map: Dict[str, CodeAngleEntry],
    *,
    filter_angle_lo: Optional[float] = None,
    filter_angle_hi: Optional[float] = None,
    angle_period_deg: float = 360.0,
    show_start_bit_index: bool = False,
) -> None:
    """
    Печатает таблицу кодовой карты.

    Сортировка и фильтрация выполняются по angle_center_deg,
    то есть в единицах угла, а не индекса.
    """
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

    entries = sorted(code_angle_map.values(), key=lambda e: e.angle_center_deg)

    for entry in entries:
        if filter_angle_lo is not None and filter_angle_hi is not None:
            if not angle_in_range_deg(
                entry.angle_center_deg,
                filter_angle_lo,
                filter_angle_hi,
                angle_period_deg=angle_period_deg,
            ):
                continue

        if show_start_bit_index:
            print(
                f"  {entry.angle_lo_deg:>13.4f} | "
                f"{entry.angle_center_deg:>14.4f} | "
                f"{entry.angle_hi_deg:>14.4f} | "
                f"{entry.start_bit_index:>8d} | "
                f"{entry.codeword:>{w}}"
            )
        else:
            print(
                f"  {entry.angle_lo_deg:>13.4f} | "
                f"{entry.angle_center_deg:>14.4f} | "
                f"{entry.angle_hi_deg:>14.4f} | "
                f"{entry.codeword:>{w}}"
            )

    print(sep)


# ---------------------------------------------------------------------------
# Основная функция оценки угла
# ---------------------------------------------------------------------------

def estimate_disk_angle_from_result(
    detection_result: "BitRecoveryResult",
    code_angle_map: Dict[str, CodeAngleEntry],
    codeword_length: int,
    total_code_bits: int,
    sensor_center_px: float,
    angle_per_px_deg: float,
    angle_period_deg: float = 360.0,
) -> AngleEstimationResult:
    """
    Оценка абсолютного угла диска по видимому участку кода.

    Логика:
    -------
    1. Из recovered_bits извлекаются валидные биты.
    2. По всем валидным битам вычисляется единый агрегированный центр
       видимого участка кода на сенсоре:
           x_vis = sum(w_i * x_i) / sum(w_i)
    3. Каждое совпавшее кодовое окно длины m даёт:
           phi_win = angle_center_deg(codeword)
       после чего вычисляется угол центра ВСЕГО видимого участка:
           phi_vis = phi_win + (n_bits/2 - s - m/2) * delta_phi
    4. По каждой такой привязке строится оценка угла диска:
           theta_s = phi_vis - angle_per_px_deg * (x_vis - sensor_center_px)
    5. Финальный ответ — circular mean / std по всем theta_s.

    Важный смысл:
    --------------
    Геометрия изображения оценивается по всему видимому коду один раз.
    Де-брейновские окна используются только для абсолютной фазовой привязки
    этого видимого фрагмента на диске.

    Parameters
    ----------
    detection_result : BitRecoveryResult
    code_angle_map : Dict[str, CodeAngleEntry]
        Угловая карта кодовых слов.
    codeword_length : int
        Длина кодового слова m.
    total_code_bits : int
        Число бит на полном диске N.
    sensor_center_px : float
        Опорный центр сенсора.
    angle_per_px_deg : float
        Масштаб [град/пиксель] со знаком.
    angle_period_deg : float
        Полный угловой период.

    Returns
    -------
    AngleEstimationResult
    """
    if codeword_length <= 0:
        raise ValueError("codeword_length must be > 0")
    if total_code_bits <= 0:
        raise ValueError("total_code_bits must be > 0")
    if angle_period_deg <= 0:
        raise ValueError("angle_period_deg must be > 0")

    if code_angle_map:
        sample_key = next(iter(code_angle_map.keys()))
        map_codeword_length = len(sample_key)
        if map_codeword_length != codeword_length:
            raise ValueError(
                f"codeword_length={codeword_length} does not match "
                f"map word length={map_codeword_length}"
            )

    # ------------------------------------------------------------------
    # 1. Извлекаем валидные recovered_bits
    # ------------------------------------------------------------------
    bit_values, bit_centers_px, bit_widths_px = _extract_valid_recovered_bits_geometry(
        detection_result
    )

    n_bits = len(bit_values)
    total_windows = max(0, n_bits - codeword_length + 1)

    result = AngleEstimationResult(
        codeword_length=codeword_length,
        visible_bits=n_bits,
        total_windows=total_windows,
        matched_windows=0,
    )

    if n_bits == 0:
        return result

    # ------------------------------------------------------------------
    # 2. Единый центр всего видимого участка в пикселях
    # ------------------------------------------------------------------
    visible_code_center_px, visible_code_span_px = _compute_visible_code_center_px(
        bit_centers_px,
        bit_widths_px,
    )
    result.visible_code_center_px = visible_code_center_px
    result.visible_code_span_px = visible_code_span_px

    if total_windows == 0 or visible_code_center_px is None:
        return result

    # ------------------------------------------------------------------
    # 3. Для каждого matched window вычисляем угол центра ВСЕГО видимого кода
    # ------------------------------------------------------------------
    delta_phi = angle_period_deg / total_code_bits

    samples: List[AngleWindowSample] = []
    angle_values_deg: List[float] = []

    for s in range(total_windows):
        codeword = "".join(str(bit_values[s + k]) for k in range(codeword_length))

        entry = code_angle_map.get(codeword)
        if entry is None:
            continue

        phi_win = entry.angle_center_deg

        # Смещение от центра matched-window к центру всего видимого фрагмента.
        # Всё вычисляется в битовых шагах, затем переводится в градусы.
        offset_bits = (n_bits / 2.0) - s - (codeword_length / 2.0)
        phi_vis = wrap_angle_deg(
            phi_win + offset_bits * delta_phi,
            angle_period_deg,
        )

        theta_s = wrap_angle_deg(
            phi_vis - angle_per_px_deg * (visible_code_center_px - sensor_center_px),
            angle_period_deg,
        )

        angle_values_deg.append(theta_s)
        samples.append(
            AngleWindowSample(
                window_start=s,
                codeword=codeword,
                matched_window_angle_deg=phi_win,
                visible_code_angle_deg=phi_vis,
                visible_code_center_px=visible_code_center_px,
                estimated_angle_deg=theta_s,
                start_bit_index=entry.start_bit_index,
            )
        )

    result.matched_windows = len(samples)
    result.samples = samples

    if not angle_values_deg:
        return result

    angles_arr = np.asarray(angle_values_deg, dtype=np.float64)

    result.mean_angle_deg = circular_mean_deg(
        angles_arr,
        angle_period_deg=angle_period_deg,
    )
    result.std_angle_deg = circular_std_deg(
        angles_arr,
        angle_period_deg=angle_period_deg,
    )

    return result


if __name__ == "__main__":
    code_map = build_code_angle_map(
        FULL_DISK_CODE_SEQUENCE,
        TOTAL_CODE_BITS_ON_DISK,
        CODEWORD_LENGTH_BITS,
        angle_period_deg=ANGLE_PERIOD_DEG,
    )

    print("Фильтр по углу центра окна: 90°..95°")
    print_code_angle_table(
        code_map,
        filter_angle_lo=90.0,
        filter_angle_hi=95.0,
        angle_period_deg=ANGLE_PERIOD_DEG,
        show_start_bit_index=False,
    )