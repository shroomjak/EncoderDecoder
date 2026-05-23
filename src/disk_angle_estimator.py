"""
disk_angle_estimator.py — расчёт абсолютного угла диска по кодовой дорожке
абсолютного энкодера с де-Брейновской подпоследовательностью.

Алгоритм (estimate_disk_angle_from_result)
------------------------------------------
Шаг 1. Масштаб angle_per_px = (angle_period / N) / mean_period_px
Шаг 2. Извлечение валидных бит, центров и ширин.
Шаг 3. Скользящие окна: lookup + theta = phi_center - k*(x_center - x0)
Шаг 4. Вес виньетирования: w(u) = 1 - k*(1 - cos(u*pi/2)^s)
Шаг 5. Weighted circular mean / std
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


"""
new code lim 7
FULL_DISK_CODE_SEQUENCE = (
    "0101010101101010100101010001010110010101110101011110101001101010000101001"
    "0010100111010110110101100010100011010111001010000010110100101101110100100"
    "0101100110100110010110000101110110100010010111000101111001011111010111111"
    "0100111101000111010000110100000010010011011011001001000010011000100111001"
    "0011111011011110110011101100011011000001000100011001000111101110111001101"
    "11000010000111011110001000001101111100111000000"
)
"""

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
    weight: float = 1.0
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
# Углы
# ---------------------------------------------------------------------------

def wrap_angle_deg(angle_deg: float, angle_period_deg: float = 360.0) -> float:
    return float(angle_deg % angle_period_deg)


def angle_in_range_deg(
    angle_deg: float, lo_deg: float, hi_deg: float,
    angle_period_deg: float = 360.0,
) -> bool:
    a = angle_deg % angle_period_deg
    lo = lo_deg % angle_period_deg
    hi = hi_deg % angle_period_deg
    if lo <= hi:
        return lo <= a < hi
    return a >= lo or a < hi


def circular_mean_deg(
    angles_deg: np.ndarray,
    weights: Optional[np.ndarray] = None,
    angle_period_deg: float = 360.0,
) -> Optional[float]:
    if angles_deg is None or len(angles_deg) == 0:
        return None
    a = np.asarray(angles_deg, dtype=np.float64)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=np.float64)
    wsum = w.sum()
    if wsum <= 1e-12:
        return None
    phase = a * (2.0 * np.pi / angle_period_deg)
    s = np.dot(w, np.sin(phase))
    c = np.dot(w, np.cos(phase))
    if abs(s) <= 1e-12 and abs(c) <= 1e-12:
        return None
    mp = np.arctan2(s, c)
    if mp < 0:
        mp += 2.0 * np.pi
    return float(mp * angle_period_deg / (2.0 * np.pi)) % angle_period_deg


def circular_std_deg(
    angles_deg: np.ndarray,
    weights: Optional[np.ndarray] = None,
    angle_period_deg: float = 360.0,
) -> Optional[float]:
    if angles_deg is None or len(angles_deg) == 0:
        return None
    a = np.asarray(angles_deg, dtype=np.float64)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=np.float64)
    wsum = w.sum()
    if wsum <= 1e-12:
        return None
    phase = a * (2.0 * np.pi / angle_period_deg)
    s = np.dot(w, np.sin(phase))
    c = np.dot(w, np.cos(phase))
    r = min(np.hypot(s, c) / wsum, 1.0 - 1e-15)
    if r <= 1e-12:
        return None
    return float(np.sqrt(-2.0 * np.log(r)) * angle_period_deg / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Кодовая карта
# ---------------------------------------------------------------------------

def build_code_angle_map(
    code_sequence: str,
    total_code_bits: int,
    codeword_length: int,
    angle_period_deg: float = 360.0,
) -> Dict[str, dict]:
    """
    codeword -> {start_bit_index, angle_lo_deg, angle_center_deg, angle_hi_deg}.
    Углы линейные (без wrap). Циклический обход через удвоенную строку.
    """
    seq = code_sequence.strip()
    if len(seq) != total_code_bits:
        raise ValueError(f"len={len(seq)} != {total_code_bits}")
    if codeword_length <= 0 or codeword_length > total_code_bits:
        raise ValueError("invalid codeword_length")

    delta = angle_period_deg / total_code_bits
    half_m = codeword_length * 0.5

    # Удвоенная строка для циклического slice без %
    doubled = seq + seq
    code_map: Dict[str, dict] = {}
    duplicates: Dict[str, List[int]] = {}

    for si in range(total_code_bits):
        cw = doubled[si: si + codeword_length]
        if cw in code_map:
            duplicates.setdefault(cw, [code_map[cw]["start_bit_index"]]).append(si)
        else:
            code_map[cw] = {
                "start_bit_index": si,
                "angle_lo_deg": si * delta,
                "angle_center_deg": (si + half_m) * delta,
                "angle_hi_deg": (si + codeword_length) * delta,
            }

    if duplicates:
        details = "; ".join(f"{cw}: {v}" for cw, v in list(duplicates.items())[:8])
        raise ValueError(f"Дубликаты: {details}")
    return code_map


def print_code_angle_table(
    code_angle_map: Dict[str, dict],
    filter_angle_lo: Optional[float] = None,
    filter_angle_hi: Optional[float] = None,
    angle_period_deg: float = 360.0,
    show_start_bit_index: bool = False,
) -> None:
    if not code_angle_map:
        print("(empty)")
        return

    w = len(next(iter(code_angle_map)))
    idx_hdr = f"{'StartIdx':>8} | " if show_start_bit_index else ""
    header = f"  {'lo°':>13} | {'center°':>14} | {'hi°':>14} | {idx_hdr}{'Код':>{w}}"
    sep = "-" * len(header)
    print(sep); print(header); print(sep)

    for cw, e in sorted(code_angle_map.items(),
                         key=lambda kv: kv[1]["angle_center_deg"] % angle_period_deg):
        lo_w = e["angle_lo_deg"] % angle_period_deg
        c_w = e["angle_center_deg"] % angle_period_deg
        hi_w = e["angle_hi_deg"] % angle_period_deg
        if filter_angle_lo is not None and filter_angle_hi is not None:
            if not angle_in_range_deg(c_w, filter_angle_lo, filter_angle_hi, angle_period_deg):
                continue
        idx_s = f"{e['start_bit_index']:>8d} | " if show_start_bit_index else ""
        print(f"  {lo_w:>13.4f} | {c_w:>14.4f} | {hi_w:>14.4f} | {idx_s}{cw:>{w}}")
    print(sep)


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def estimate_disk_angle_from_result(
    detection_result: "BitRecoveryResult",
    code_sequence: str,
    total_code_bits: int,
    codeword_length: int,
    sensor_center_px: float,
    sensor_width_px: float,
    angle_period_deg: float = 360.0,
    vignetting_k: float = 0.75,
    vignetting_s: float = 0.5,
) -> AngleEstimationResult:
    m = codeword_length

    # --- Шаг 1. Масштаб ---
    total_px = 0.0
    total_n = 0
    for seg in detection_result.bit_segments:
        n = int(seg.n_bits)
        p = float(seg.measured_period)
        if n > 0 and p > 1e-12:
            total_px += p * n
            total_n += n

    mean_period_px = (total_px / total_n) if total_n > 0 else None

    result = AngleEstimationResult(
        codeword_length=m, visible_bits=0, total_windows=0, matched_windows=0,
        mean_bit_period_px=mean_period_px, reverse_direction=REVERSE_AXIS_SIGN,
    )

    if mean_period_px is None or mean_period_px <= 1e-12:
        return result

    delta_phi = angle_period_deg / total_code_bits
    angle_per_px = delta_phi / mean_period_px
    result.angle_per_px_deg = angle_per_px

    # --- Карта ---
    eff_seq = code_sequence[::-1] if REVERSE_AXIS_SIGN else code_sequence
    code_map = build_code_angle_map(eff_seq, total_code_bits, m, angle_period_deg)

    # --- Шаг 2. Биты ---
    bit_values: List[int] = []
    bit_centers_px: List[float] = []
    bit_widths_px: List[float] = []

    for rb in sorted(detection_result.recovered_bits, key=lambda b: b.position):
        if not (0 <= rb.segment_idx < len(detection_result.bit_segments)):
            continue
        seg = detection_result.bit_segments[rb.segment_idx]
        pp = float(seg.measured_period)
        if pp <= 1e-12:
            continue
        bs = float(rb.position)
        be = bs + pp
        eps = max(1e-9, 1e-6 * pp)
        if bs < float(seg.start_pos) - eps or be > float(seg.end_pos) + eps:
            continue
        bit_values.append(int(rb.value))
        bit_centers_px.append(bs + 0.5 * pp)
        bit_widths_px.append(pp)

    n_bits = len(bit_values)
    n_win = max(0, n_bits - m + 1)
    result.visible_bits = n_bits
    result.total_windows = n_win

    if n_win == 0:
        return result

    centers = np.asarray(bit_centers_px, dtype=np.float64)
    widths = np.asarray(bit_widths_px, dtype=np.float64)

    # --- Центры окон через cumsum (O(n) вместо O(n*m)) ---
    wc = widths * centers
    cum_wc = np.empty(n_bits + 1, dtype=np.float64)
    cum_w = np.empty(n_bits + 1, dtype=np.float64)
    cum_wc[0] = 0.0; cum_w[0] = 0.0
    np.cumsum(wc, out=cum_wc[1:])
    np.cumsum(widths, out=cum_w[1:])

    sum_wc = cum_wc[m:m + n_win] - cum_wc[:n_win]
    sum_w = cum_w[m:m + n_win] - cum_w[:n_win]
    valid_w = sum_w > 1e-12
    safe_w = np.where(valid_w, sum_w, 1.0)
    x_win_all = sum_wc / safe_w  # центры всех окон

    # --- Шаг 3. Lookup ---
    bit_chars = "".join(str(v) for v in bit_values)

    matched_s: List[int] = []
    matched_phi: List[float] = []
    matched_sbi: List[int] = []
    matched_cw: List[str] = []

    for s in range(n_win):
        if not valid_w[s]:
            continue
        cw = bit_chars[s:s + m]
        entry = code_map.get(cw)
        if entry is not None:
            matched_s.append(s)
            matched_phi.append(entry["angle_center_deg"])
            matched_sbi.append(entry["start_bit_index"])
            matched_cw.append(cw)

    n_matched = len(matched_s)
    if n_matched == 0:
        return result

    idx_arr = np.array(matched_s, dtype=np.intp)
    phi_arr = np.array(matched_phi, dtype=np.float64)
    x_centers = x_win_all[idx_arr]

    # theta = phi - k * (x - x0), затем wrap
    theta_arr = (phi_arr - angle_per_px * (x_centers - sensor_center_px)) % angle_period_deg

    # --- Шаг 4. Виньетирование (векторное) ---
    sensor_half_w = sensor_width_px * 0.5
    u = (x_centers - sensor_center_px) / sensor_half_w if sensor_half_w > 1e-12 else np.full(n_matched, 2.0)
    inside = np.abs(u) <= 1.0

    if vignetting_k <= 1e-15:
        w_vign = np.where(inside, 1.0, 0.0)
    else:
        cos_val = np.cos(u * (np.pi * 0.5))
        cos_pow = np.abs(cos_val) ** vignetting_s if vignetting_s > 1e-15 else np.ones(n_matched)
        w_vign = np.where(inside, np.maximum(0.0, 1.0 - vignetting_k * (1.0 - cos_pow)), 0.0)

    alive = w_vign > 1e-15

    # --- Сборка samples ---
    samples: List[AngleWindowSample] = []
    for i in range(n_matched):
        if alive[i]:
            samples.append(AngleWindowSample(
                window_start=int(idx_arr[i]),
                codeword=matched_cw[i],
                window_center_px=float(x_centers[i]),
                window_angle_deg=float(phi_arr[i] % angle_period_deg),
                estimated_angle_deg=float(theta_arr[i]),
                weight=float(w_vign[i]),
                start_bit_index=matched_sbi[i],
            ))

    result.samples = samples
    result.matched_windows = len(samples)

    if not samples:
        return result

    # --- Шаг 5. Weighted circular mean / std ---
    final_angles = theta_arr[alive]
    final_weights = w_vign[alive]

    result.mean_angle_deg = circular_mean_deg(final_angles, weights=final_weights, angle_period_deg=angle_period_deg)
    result.std_angle_deg = circular_std_deg(final_angles, weights=final_weights, angle_period_deg=angle_period_deg)

    return result


if __name__ == "__main__":
    code_map = build_code_angle_map(
        FULL_DISK_CODE_SEQUENCE[::-1],
        TOTAL_CODE_BITS_ON_DISK,
        CODEWORD_LENGTH_BITS,
        ANGLE_PERIOD_DEG,
    )
    print("Фильтр 226°..230°")
    print_code_angle_table(code_map, 226, 230, ANGLE_PERIOD_DEG, True)