"""
disk_angle_estimator.py — упрощённый расчёт абсолютного угла диска
по полной видимой битовой последовательности внутри ROI.

Логика:
1. Из результата blais_rioux берутся только ПОЛНЫЕ биты (recovered_bits).
2. По скользящим окнам длины m ищутся совпадения с кодовой картой.
3. Каждое совпавшее окно голосует за абсолютное смещение:
       offset = disk_start_index - roi_window_start
4. Берётся offset с максимальным числом голосов.
5. По этому offset каждому полному биту ставится в соответствие абсолютный
   индекс на диске.
6. Абсолютный угол диска оценивается по всем полным битам:
       theta_i = phi_i - k * (x_i - x_center)
   и затем усредняется через circmean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.blais_rioux import EdgeDetectionResult


@dataclass
class AngleBitSample:
    """Оценка угла по одному полному биту."""
    roi_bit_index: int
    disk_bit_index: int
    bit_value: int
    bit_center_px: float
    disk_bit_angle_deg: float
    estimated_angle_deg: float


@dataclass
class AngleEstimationResult:
    """Результат оценки угла по полным битам внутри ROI."""
    codeword_length: int
    visible_bits: int
    total_windows: int
    matched_windows: int
    chosen_offset: Optional[int] = None
    samples: List[AngleBitSample] = field(default_factory=list)
    mean_angle_deg: Optional[float] = None
    std_angle_deg: Optional[float] = None


def wrap_angle_deg(angle_deg: float, angle_period_deg: float = 360.0) -> float:
    """Нормализация угла в диапазон [0, angle_period_deg)."""
    if angle_period_deg <= 0:
        raise ValueError("angle_period_deg must be > 0")
    return float(angle_deg % angle_period_deg)


def circular_mean_deg(
    angles_deg: np.ndarray,
    weights: Optional[np.ndarray] = None,
    angle_period_deg: float = 360.0,
) -> Optional[float]:
    """
    Circular mean для углов в градусах.
    """
    if angles_deg is None or len(angles_deg) == 0:
        return None

    angles_deg = np.asarray(angles_deg, dtype=np.float64)

    if weights is None:
        weights = np.ones_like(angles_deg, dtype=np.float64)
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
    """
    Circular std для углов в градусах.
    """
    if angles_deg is None or len(angles_deg) == 0:
        return None

    angles_deg = np.asarray(angles_deg, dtype=np.float64)

    if weights is None:
        weights = np.ones_like(angles_deg, dtype=np.float64)
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

    r = np.hypot(s, c) / wsum
    r = float(np.clip(r, 0.0, 1.0))

    if r <= 1e-12:
        return None

    std_phase = np.sqrt(max(0.0, -2.0 * np.log(r)))
    return float(std_phase * angle_period_deg / (2.0 * np.pi))


def _validate_binary_code_sequence(code_sequence: str, total_code_bits: int) -> str:
    """
    Проверка полной бинарной кодовой последовательности.
    """
    if total_code_bits <= 0:
        raise ValueError("total_code_bits must be > 0")

    if code_sequence is None:
        raise ValueError("code_sequence is None")

    seq = code_sequence.strip()

    if len(seq) != total_code_bits:
        raise ValueError(
            f"len(code_sequence)={len(seq)} does not match total_code_bits={total_code_bits}"
        )

    if any(ch not in "01" for ch in seq):
        raise ValueError("code_sequence must contain only '0' and '1'")

    return seq


def build_code_angle_map(
    code_sequence: str,
    total_code_bits: int,
    codeword_length: int,
) -> Dict[str, int]:
    """
    Строит карту:
        codeword -> start_bit_index

    Кодовая последовательность считается циклической.
    Для полной дорожки длины N строится N окон длины m.

    ВАЖНО:
    функция требует, чтобы все циклические кодовые слова длины m были уникальны.
    Иначе абсолютная синхронизация по одному окну неоднозначна.

    Parameters
    ----------
    code_sequence : str
        Полная циклическая кодовая последовательность длины N.
    total_code_bits : int
        Число бит на диске, N.
    codeword_length : int
        Длина кодового слова, m.

    Returns
    -------
    Dict[str, int]
        Словарь {кодовое_слово: стартовый_индекс_на_диске}.
    """
    seq = _validate_binary_code_sequence(code_sequence, total_code_bits)

    if codeword_length <= 0:
        raise ValueError("codeword_length must be > 0")
    if codeword_length > total_code_bits:
        raise ValueError(
            f"codeword_length={codeword_length} must be <= total_code_bits={total_code_bits}"
        )

    code_map: Dict[str, int] = {}
    duplicates: Dict[str, List[int]] = {}

    for start_idx in range(total_code_bits):
        codeword = "".join(
            seq[(start_idx + k) % total_code_bits]
            for k in range(codeword_length)
        )

        if codeword in code_map:
            duplicates.setdefault(codeword, [code_map[codeword]]).append(start_idx)
        else:
            code_map[codeword] = start_idx

    if duplicates:
        details = []
        for codeword, starts in duplicates.items():
            details.append(f"{codeword}: starts={starts}")
        details_str = "; ".join(details[:8])
        if len(details) > 8:
            details_str += "; ..."
        raise ValueError(
            "Кодовая карта неоднозначна: некоторые кодовые слова длины m "
            f"встречаются более одного раза. {details_str}"
        )

    return code_map


def estimate_disk_angle_from_result(
    detection_result: "EdgeDetectionResult",
    code_angle_map: Dict[str, int],
    codeword_length: int,
    total_code_bits: int,
    sensor_center_px: float,
    angle_per_px_deg: float,
    angle_period_deg: float = 360.0,
) -> AngleEstimationResult:
    """
    Оценка абсолютного угла диска.

    Используются только полные биты из detection_result.recovered_bits.

    Формулы:
    --------
    1) Угол центра бита с индексом j на диске:
           phi_j = (j + 0.5) * angle_period / total_code_bits

    2) Если окно длины m, начинающееся в ROI с индекса s, соответствует
       слову со стартовым индексом J на диске, то абсолютный индекс любого
       видимого бита i:
           j_i = (J - s + i) mod N

    3) Оценка абсолютного угла диска по биту:
           theta_i = phi_j - angle_per_px_deg * (x_i - sensor_center_px)

       Здесь angle_per_px_deg включает знак направления.

    4) Итог:
           theta = circmean(theta_i)
           std   = circstd(theta_i)

    Parameters
    ----------
    detection_result : EdgeDetectionResult
        Результат blais_rioux.
    code_angle_map : Dict[str, int]
        Карта {кодовое_слово: стартовый_индекс_на_диске}.
    codeword_length : int
        Длина кодового слова m.
    total_code_bits : int
        Полная длина кода N.
    sensor_center_px : float
        Центр ПЗС в пикселях.
    angle_per_px_deg : float
        Коэффициент k в град/пикс, включая знак направления.
    angle_period_deg : float
        Период угла, обычно 360.

    Returns
    -------
    AngleEstimationResult
        Средний угол и circular std.
    """
    if codeword_length <= 0:
        raise ValueError("codeword_length must be > 0")
    if total_code_bits <= 0:
        raise ValueError("total_code_bits must be > 0")
    if angle_period_deg <= 0:
        raise ValueError("angle_period_deg must be > 0")

    # Берём только полные биты, которые уже отфильтрованы в blais_rioux.
    recovered_bits = sorted(detection_result.recovered_bits, key=lambda b: b.position)

    bit_values: List[int] = []
    bit_centers_px: List[float] = []

    for rb in recovered_bits:
        if rb.segment_idx < 0 or rb.segment_idx >= len(detection_result.bit_segments):
            continue

        seg = detection_result.bit_segments[rb.segment_idx]
        period_px = float(seg.measured_period)

        if period_px <= 1e-12:
            continue

        bit_start = float(rb.position)
        bit_end = bit_start + period_px

        # Защита от возможной несогласованности:
        # учитываем только действительно полный бит внутри сегмента.
        eps = max(1e-9, 1e-6 * period_px)
        if bit_start < float(seg.start_pos) - eps:
            continue
        if bit_end > float(seg.end_pos) + eps:
            continue

        bit_values.append(int(rb.value))
        bit_centers_px.append(bit_start + 0.5 * period_px)

    n_bits = len(bit_values)
    total_windows = max(0, n_bits - codeword_length + 1)

    result = AngleEstimationResult(
        codeword_length=codeword_length,
        visible_bits=n_bits,
        total_windows=total_windows,
        matched_windows=0,
    )

    if n_bits < codeword_length:
        return result

    # Каждый совпавший window голосует за абсолютный offset:
    #   disk_index(i) = offset + i  (mod N)
    offset_votes: Dict[int, int] = {}

    for window_start in range(total_windows):
        codeword = "".join(
            str(bit_values[window_start + k])
            for k in range(codeword_length)
        )

        disk_start_index = code_angle_map.get(codeword)
        if disk_start_index is None:
            continue

        offset = (disk_start_index - window_start) % total_code_bits
        offset_votes[offset] = offset_votes.get(offset, 0) + 1

    if not offset_votes:
        return result

    best_vote_count = max(offset_votes.values())
    best_offsets = [off for off, cnt in offset_votes.items() if cnt == best_vote_count]

    # При равенстве голосов берём минимальный offset, чтобы результат был детерминирован.
    chosen_offset = min(best_offsets)

    result.chosen_offset = chosen_offset
    result.matched_windows = best_vote_count

    delta_phi = angle_period_deg / total_code_bits
    angle_samples: List[AngleBitSample] = []
    angle_values_deg: List[float] = []

    for roi_bit_index, (bit_value, bit_center_px) in enumerate(zip(bit_values, bit_centers_px)):
        disk_bit_index = (chosen_offset + roi_bit_index) % total_code_bits

        disk_bit_angle_deg = wrap_angle_deg(
            (disk_bit_index + 0.5) * delta_phi,
            angle_period_deg,
        )

        estimated_angle_deg = wrap_angle_deg(
            disk_bit_angle_deg - angle_per_px_deg * (bit_center_px - sensor_center_px),
            angle_period_deg,
        )

        angle_values_deg.append(estimated_angle_deg)
        angle_samples.append(AngleBitSample(
            roi_bit_index=roi_bit_index,
            disk_bit_index=disk_bit_index,
            bit_value=bit_value,
            bit_center_px=bit_center_px,
            disk_bit_angle_deg=disk_bit_angle_deg,
            estimated_angle_deg=estimated_angle_deg,
        ))

    result.samples = angle_samples

    angles_arr = np.array(angle_values_deg, dtype=np.float64)
    result.mean_angle_deg = circular_mean_deg(
        angles_arr,
        angle_period_deg=angle_period_deg,
    )
    result.std_angle_deg = circular_std_deg(
        angles_arr,
        angle_period_deg=angle_period_deg,
    )

    return result