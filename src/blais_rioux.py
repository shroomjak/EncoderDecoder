"""
blais_rioux.py — Модуль обработки сигнала методом Blais-Rioux

Алгоритм:
1. Глобальная нормализация сигнала
2. Гауссово сглаживание (подавление шума)
3. КИХ фильтр D1 (моментное ядро Blais-Rioux): h[k] = k / Σk²
4. КИХ фильтр D2 = D1(D1(signal))
5. Поиск нулей D2 с линейной интерполяцией: pos = x_A + A/(A-B)
6. Определение ROI по детектированным фронтам
7. Минимаксная нормализация внутри ROI
8. Восстановление бит по знаку D1 на фронтах:
   - D1 > 0 → сигнал возрастает → переход 0→1 → после фронта бит = 1
   - D1 < 0 → сигнал убывает → переход 1→0 → после фронта бит = 0
9. Измерение периода: для каждого сегмента period_i = distance_i / nBits_i
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, correlate1d


@dataclass
class BRConfig:
    """Конфигурация BR."""
    
    filter_order: int = 2
    """Порядок КИХ фильтра N (длина ядра 2N+1): 2 или 4."""
    
    peak_threshold_rel: float = 0.15
    """Порог |D1| относительно максимума (0..1)."""
    
    min_edge_distance_factor: float = 0.3
    """Минимальное расстояние между фронтами как доля от bit_width_px."""
    
    bit_width_px: float = 5
    """Априорная ширина бита в пикселях."""
    
    smoothing_sigma: float = 0.5
    """Cглаживания перед дифференцированием."""
    
    minmax_window_px: int = 50
    """Ширина окна для скользящего min/max нормирования."""


@dataclass
class DetectedEdge:
    """Детектированный фронт."""
    
    position: float
    """Субпиксельная позиция фронта."""
    
    d1_value: float
    """Значение D1 в этой точке (знак = направление перехода)."""


@dataclass
class BitSegment:
    """Сегмент восстановленных бит между двумя фронтами."""
    
    start_pos: float
    """Позиция начального фронта (субпиксельная)."""
    
    end_pos: float
    """Позиция конечного фронта (субпиксельная)."""
    
    bit_value: int
    """Значение бита: 0 или 1."""
    
    n_bits: int
    """Количество бит в сегменте."""
    
    distance: float
    """Расстояние между фронтами в пикселях."""
    
    measured_period: float
    """Измеренный период для этого сегмента: distance / n_bits."""


@dataclass  
class RecoveredBit:
    """Восстановленный бит с координатой."""
    
    value: int
    """Значение бита: 0 или 1."""
    
    position: float
    """Позиция начала бита (интерполированная)."""
    
    segment_idx: int
    """Индекс сегмента, которому принадлежит бит."""


@dataclass
class EdgeDetectionResult:
    """Результат детектирования фронтов и восстановления бит."""
    
    first_derivative: np.ndarray
    """Первая производная D1."""
    
    second_derivative: np.ndarray
    """Вторая производная D2."""
    
    smoothed_signal: np.ndarray
    """Сглаженный сигнал."""
    
    local_norm_signal: np.ndarray
    """Локально нормированный сигнал (minmax в ROI)."""
    
    detected_edges: List[DetectedEdge]
    """Список детектированных фронтов."""
    
    bit_width_px: float
    """Априорная ширина бита."""
    
    measured_bit_period: float
    """Средний измеренный период."""
    
    bit_segments: List[BitSegment]
    """Сегменты восстановленных бит."""
    
    recovered_bits: List[RecoveredBit]
    """Восстановленные биты с координатами."""
    
    recovered_bit_values: np.ndarray
    """Плоский массив значений восстановленных бит."""
    
    roi_start: float
    """Начало ROI."""
    
    roi_end: float
    """Конец ROI."""
    
    bit_errors: int
    """Количество ошибок в битах."""
    
    edge_errors: np.ndarray
    """Ошибки позиционирования фронтов."""
    
    rms_edge_error: float
    """RMS ошибки позиционирования."""
    
    peak_threshold: float
    """Использованный порог амплитуды."""
    
    accuracy: float
    """Точность восстановления (%)."""
    
    config: BRConfig = field(repr=False)
    """Конфигурация."""


def make_br_kernel(order: int) -> np.ndarray:
    """
    Создаёт КИХ ядро BR для первой производной.
    
    h[k] = k / sum|k|, k in {-1, 1}
    
    Моментный фильтр, дающий положительное значение при
    возрастании сигнала и отрицательное при убывании.
    
    Parameters
    ----------
    order : int
        Порядок фильтра.

    Returns
    -------
    np.ndarray
        Ядро фильтра длиной 2*order + 1.
    """
    k = np.array([-1] * order + [0] + [1] * order)
    norm_l1 = np.sum(abs(k))
    return k / norm_l1


def correlate_mirror(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Корреляция сигнала с ядром (зеркальные граничные условия).

    Parameters
    ----------
    signal : np.ndarray
        Входной сигнал.
    kernel : np.ndarray
        Ядро.
        
    Returns
    -------
    np.ndarray
        Результат корреляции той же длины, что и signal.
    """
    return correlate1d(signal, kernel, mode='reflect')


def normalize_global(signal: np.ndarray) -> np.ndarray:
    """
    Глобальная нормализация сигнала к диапазону [0, 1].
    
    Parameters
    ----------
    signal : np.ndarray
        Входной сигнал.
        
    Returns
    -------
    np.ndarray
        Нормализованный сигнал.
    """
    s_min, s_max = signal.min(), signal.max()
    return (signal - s_min) / (s_max - s_min + 1e-12)


def apply_minmax_normalization(
    signal: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Минимаксная нормализация по скользящему окну внутри ROI.
    
    normalized[i] = (signal[i] - global_min) / (running_max[i] - global_min)
    
    Parameters
    ----------
    signal : np.ndarray
        Входной сигнал.
    window_size : int
        Ширина скользящего окна.
        
    Returns
    -------
    np.ndarray
        Нормализованный сигнал.
    """
    n = len(signal)
    
    # Скользящие min/max
    global_min = min(signal)
    running_max = maximum_filter1d(signal, size=window_size, mode='reflect')
    
    # Нормализация только внутри ROI
    normalized = signal.copy()
    for i in range(len(normalized)):
        local_max = max(running_max[i], 0.5)
        range_val = local_max - global_min + 1e-12
        normalized[i] = (signal[i] - global_min) / range_val
    
    return normalized


def find_zero_crossings_d2(
    d2: np.ndarray,
    d1: np.ndarray
) -> List[DetectedEdge]:
    """
    Поиск нулей D2 с линейной интерполяцией.
    
    Формула: pos = i + A / (A - B), где A = D2[i], B = D2[i+1], A*B < 0.
    
    Нули D2 соответствуют экстремумам D1, т.е. позициям фронтов.
    
    Parameters
    ----------
    d2 : np.ndarray
        Вторая производная.
    d1 : np.ndarray
        Первая производная.
        
    Returns
    -------
    List[DetectedEdge]
        Список детектированных фронтов.
    """
    edges = []
    n = len(d2)
    
    for i in range(n - 1):
        left = d2[i]
        right = d2[i + 1]

        if left * right < 0:
            pos = i + left / (left - right)
            
            # Интерполяция D1 в этой точке
            frac = pos - i
            d1_at_pos = d1[i] * (1 - frac) + d1[i + 1] * frac
            
            edges.append(DetectedEdge(
                position=pos,
                d1_value=d1_at_pos,
            ))
    
    return edges


def filter_by_min_distance(
    edges: List[DetectedEdge],
    min_dist: float
) -> List[DetectedEdge]:
    """
    Фильтрация фронтов по минимальному расстоянию.
    
    При конфликте оставляем фронт с большей амплитудой |D1|.
    
    Parameters
    ----------
    edges : List[DetectedEdge]
        Список фронтов.
    min_dist : float
        Минимальное допустимое расстояние.
        
    Returns
    -------
    List[DetectedEdge]
        Отфильтрованный список.
    """
    if not edges:
        return []
    
    sorted_edges = sorted(edges, key=lambda e: e.position)
    result = [sorted_edges[0]]
    
    for curr in sorted_edges[1:]:
        prev = result[-1]
        dist = curr.position - prev.position
        
        if dist < min_dist:
            # Оставляем более сильный фронт
            if abs(curr.d1_value) > abs(prev.d1_value):
                result[-1] = curr
        else:
            result.append(curr)
    
    return result


def build_bit_segments(
    edges: List[DetectedEdge],
    bit_width_px: float,
    roi_end: float
) -> List[BitSegment]:
    """
    Построение сегментов бит из фронтов.
    
    Для каждого сегмента:
    - n_bits = round(distance / bit_width_px)
    - measured_period = distance / n_bits
    - bit_value определяется по знаку D1 на начальном фронте:
      - D1 > 0 сигнал возрастает, переход 0->1, после фронта бит = 1
      - D1 < 0  сигнал убывает, переход 1->0, после фронта бит = 0
    
    Parameters
    ----------
    edges : List[DetectedEdge]
        Отфильтрованные фронты.
    bit_width_px : float
        Априорная ширина бита.
    roi_end : float
        Конец ROI.
        
    Returns
    -------
    List[BitSegment]
        Список сегментов.
    """
    if not edges:
        return []
    
    sorted_edges = sorted(edges, key=lambda e: e.position)
    segments = []
    
    for i, edge in enumerate(sorted_edges):
        next_edge = sorted_edges[i + 1] if i + 1 < len(sorted_edges) else None
        
        # Значение бита после фронта определяется знаком D1:
        bit_value = 1 if edge.d1_value > 0 else 0
        
        start_pos = edge.position
        end_pos = next_edge.position if next_edge else roi_end
        distance = end_pos - start_pos
        
        # Количество бит = округление (distance / априорная ширина)
        n_bits = max(1, round(distance / bit_width_px))
        
        # Измеренный период
        measured_period = distance / n_bits
        
        segments.append(BitSegment(
            start_pos=start_pos,
            end_pos=end_pos,
            bit_value=bit_value,
            n_bits=n_bits,
            distance=distance,
            measured_period=measured_period
        ))
    
    return segments


def extract_bits_with_positions(segments: List[BitSegment]) -> List[RecoveredBit]:
    """
    Извлечение бит с интерполированными координатами.
    
    Для каждого бита в сегменте:
    - position = start_pos + i * measured_period
    
    Parameters
    ----------
    segments : List[BitSegment]
        Сегменты бит.
        
    Returns
    -------
    List[RecoveredBit]
        Биты с координатами.
    """
    bits = []
    
    for seg_idx, seg in enumerate(segments):
        for i in range(seg.n_bits):
            # Интерполяция позиции внутри сегмента
            position = seg.start_pos + i * seg.measured_period
            bits.append(RecoveredBit(
                value=seg.bit_value,
                position=position,
                segment_idx=seg_idx
            ))
    
    return bits


def compute_mean_measured_period(segments: List[BitSegment]) -> float:
    """
    Средний измеренный период по всем сегментам.
    
    mean_period = sum(distance_i) / sum(n_bits_i)
    
    Parameters
    ----------
    segments : List[BitSegment]
        Сегменты.
        
    Returns
    -------
    float
        Средний период.
    """
    if not segments:
        return 0.0
    
    total_distance = sum(seg.distance for seg in segments)
    total_bits = sum(seg.n_bits for seg in segments)
    
    return total_distance / total_bits if total_bits > 0 else 0.0


def detect_edges_and_recover_bits(
    adc_signal: np.ndarray,
    true_bits: np.ndarray,
    true_edges: np.ndarray,
    config: Optional[BRConfig] = None
) -> EdgeDetectionResult:
    """
    Основная функция детектирования фронтов и восстановления бит.
    
    Parameters
    ----------
    adc_signal : np.ndarray
        Сигнал АЦП.
    true_bits : np.ndarray
        Истинные биты (для сравнения).
    true_edges : np.ndarray
        Истинные позиции фронтов (для сравнения).
    config : BRConfig, optional
        Конфигурация. Если None - используются значения по умолчанию.
        
    Returns
    -------
    EdgeDetectionResult
        Результат обработки.
    """
    if config is None:
        config = BRConfig()
    
    n_pixels = len(adc_signal)
    bit_width_px = config.bit_width_px
    
    # 1. Глобальная нормализация
    signal_norm = normalize_global(adc_signal.astype(np.float64))
    
    # 2. Гауссово сглаживание
    signal_smoothed = gaussian_filter1d(signal_norm, sigma=config.smoothing_sigma)

    # 3. Минимаксная нормализация внутри ROI
    local_norm_signal = apply_minmax_normalization(
        signal_smoothed, config.minmax_window_px
    )

    # 4. КИХ фильтр D1 (BR)
    br_kernel = make_br_kernel(config.filter_order)
    d1 = correlate_mirror(local_norm_signal, br_kernel)
    
    # 5. КИХ фильтр D2 = D1(D1)
    d2 = correlate_mirror(d1, br_kernel)
    
    # 6. Поиск нулей D2
    raw_edges = find_zero_crossings_d2(d2, d1)
    
    # 7. Порог по амплитуде
    max_amp = max((abs(e.d1_value) for e in raw_edges), default=0)
    peak_threshold = max_amp * config.peak_threshold_rel
    
    # 8. Фильтрация по амплитуде
    filtered_by_amp = [e for e in raw_edges if abs(e.d1_value) >= peak_threshold]
    
    # 9. Фильтрация по минимальному расстоянию
    min_dist = bit_width_px * config.min_edge_distance_factor
    filtered_edges = filter_by_min_distance(filtered_by_amp, min_dist)
    
    # 10. ROI: от первого до последнего фронта
    roi_start = 0.0
    roi_end = float(n_pixels - 1)
    if filtered_edges:
        positions = sorted(e.position for e in filtered_edges)
        roi_start = positions[0]
        roi_end = positions[-1]
    
    # 11. Построение сегментов бит
    bit_segments = build_bit_segments(filtered_edges, bit_width_px, roi_end)
    
    # 12. Измеренный средний период
    measured_bit_period = compute_mean_measured_period(bit_segments)
    
    # 13. Извлечение бит с координатами
    recovered_bits = extract_bits_with_positions(bit_segments)
    recovered_bit_values = np.array([b.value for b in recovered_bits])
    
    # 14. Сопоставление с истинными битами в ROI
    true_bit_centers = (true_edges[:-1] + true_edges[1:]) / 2.0
    margin = bit_width_px * 0.5
    
    true_bits_in_roi = []
    for i, c in enumerate(true_bit_centers):
        if roi_start - margin <= c <= roi_end + margin:
            true_bits_in_roi.append(true_bits[i])
    true_bits_in_roi = np.array(true_bits_in_roi)
    
    # 15. Подсчёт ошибок
    bit_errors = 0
    for i in range(len(true_bits_in_roi)):
        if i >= len(recovered_bit_values):
            bit_errors += 1
        elif recovered_bit_values[i] != true_bits_in_roi[i]:
            bit_errors += 1
    
    # 16. Точность
    total_bits_in_roi = len(true_bits_in_roi)
    correct_bits = total_bits_in_roi - bit_errors
    accuracy = (correct_bits / total_bits_in_roi * 100) if total_bits_in_roi > 0 else 0.0

    # 17. Ошибки позиционирования фронтов
    edge_errors = []
    used_detected = set()
    detected_positions = [e.position for e in filtered_edges]
    
    for te in true_edges:
        if te < roi_start - bit_width_px * 0.5 or te > roi_end + bit_width_px * 0.5:
            continue
        
        best_dist = float('inf')
        best_idx = -1
        
        for j, dp in enumerate(detected_positions):
            if j in used_detected:
                continue
            dist = abs(dp - te)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        
        if best_idx >= 0 and best_dist < bit_width_px * 0.5:
            edge_errors.append(detected_positions[best_idx] - te)
            used_detected.add(best_idx)
    
    edge_errors = np.array(edge_errors)
    rms_edge_error = np.sqrt(np.mean(edge_errors ** 2)) if len(edge_errors) > 0 else 0.0
    
    return EdgeDetectionResult(
        first_derivative=d1,
        second_derivative=d2,
        smoothed_signal=signal_smoothed,
        local_norm_signal=local_norm_signal,
        detected_edges=filtered_edges,
        bit_width_px=bit_width_px,
        measured_bit_period=measured_bit_period,
        bit_segments=bit_segments,
        recovered_bits=recovered_bits,
        recovered_bit_values=recovered_bit_values,
        roi_start=roi_start,
        roi_end=roi_end,
        bit_errors=bit_errors,
        edge_errors=edge_errors,
        rms_edge_error=rms_edge_error,
        peak_threshold=peak_threshold,
        accuracy=accuracy,
        config=config
    )


if __name__ == "__main__":
    # Тестовый запуск
    from ccd_simulator import simulate_ccd, SimulatorConfig
    
    sim_config = SimulatorConfig(n_bits=16, seed=42)
    sim_result = simulate_ccd(sim_config)
    
    br_config = BRConfig(bit_width_px=sim_config.bit_width_px)
    result = detect_edges_and_recover_bits(
        sim_result.adc_signal,
        sim_result.bits,
        sim_result.true_edges,
        br_config
    )
    
    print(f"Detected edges: {len(result.detected_edges)}")
    print(f"Accuracy: {result.accuracy:.1f}%")
    print(f"RMS edge error: {result.rms_edge_error:.3f} px")
    print(f"True bits:      {''.join(str(b) for b in sim_result.bits)}")
    print(f"Recovered bits: {''.join(str(b) for b in result.recovered_bit_values)}")
