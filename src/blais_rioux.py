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

from src.ccd_simulator import distort_division

@dataclass
class BRConfig:
    """Конфигурация BR."""
    
    filter_order: int = 2
    """Порядок КИХ фильтра N (длина ядра 2N+1): 2 или 4."""

    roi_start: Optional[float] = None
    """Начало ROI в пикселях. None — авто (позиция первого фронта)."""

    roi_end: Optional[float] = None
    """Конец ROI в пикселях. None — авто (позиция последнего фронта)."""

    peak_threshold_rel: float = 0.25
    """Порог |D1| относительно максимума (0..1)."""
    
    min_edge_distance_factor: float = 0.3
    """Минимальное расстояние между фронтами как доля от bit_width_px."""
    
    bit_width_px: float = 5
    """Априорная ширина бита в пикселях."""
    
    smoothing_sigma: float = 0.2
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
    """Видимая начальная позиция сегмента."""

    end_pos: float
    """Видимая конечная позиция сегмента."""

    bit_value: int
    """Значение бита: 0 или 1."""

    n_bits: int
    """Количество бит в сегменте."""

    distance: float
    """Видимая длина сегмента в пикселях."""

    measured_period: float
    """Период бита, использованный при разбиении сегмента."""

    grid_start_pos: Optional[float] = None
    """Опорная позиция начала битовой сетки для интерполяции.

    Если None, используется start_pos.
    Для левого сегмента у ROI это позволяет якорить сетку на первом фронте,
    а не на roi_start, поскольку граница ROI может разрезать бит.
    """


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

def estimate_local_distorted_bit_width(
    mid_pos: float,
    bit_width_px: float,
    n_pixels: int,
    distort_coeff: float
) -> float:
    """
    Оценка локальной ширины бита с учётом дисторсии.

    Parameters
    ----------
    mid_pos : float
        Центр сегмента в пикселях.
    bit_width_px : float
        Априорная ширина бита.
    n_pixels : int
        Длина сигнала.
    distort_coeff : float
        Коэффициент дисторсии.

    Returns
    -------
    float
        Локальная оценка ширины бита в пикселях.
    """
    center = n_pixels / 2
    pos = mid_pos + distort_coeff
    width = (
        distort_division(pos + bit_width_px, center, center, distort_coeff)
        - distort_division(pos, center, center, distort_coeff)
    )
    return max(abs(width), 1e-12)


def estimate_roi_segment_n_bits(
    visible_distance: float,
    reference_period: float
) -> int:
    """
    Оценка количества бит в сегменте, примыкающем к ROI.

    Для крайних сегментов нельзя использовать round(distance / T), потому что
    граница ROI может проходить внутри бита. В таком случае видимая длина
    сегмента равна:

        visible_distance = N * T - delta,  0 <= delta < T

    где N — число бит, пересекающих ROI. Поэтому нужно использовать ceil.

    Parameters
    ----------
    visible_distance : float
        Видимая длина сегмента внутри ROI.
    reference_period : float
        Средняя оценка ширины бита.

    Returns
    -------
    int
        Количество бит, пересекающих ROI в данном сегменте.
    """
    if visible_distance <= 1e-12:
        return 0

    ref = max(reference_period, 1e-12)
    return max(1, int(np.ceil((visible_distance - 1e-9) / ref)))


def build_internal_bit_segments(
    edges: List[DetectedEdge],
    bit_width_px: float,
    n_pixels: int,
    distort_coeff: float
) -> List[BitSegment]:
    """
    Построение внутренних сегментов между соседними фронтами.

    Эти сегменты ограничены реальными фронтами с обеих сторон, поэтому для них
    число бит можно оценивать стандартно:
    - локальная ширина бита -> с учётом дисторсии
    - n_bits = round(distance / local_bit_width)
    """
    if len(edges) < 2:
        return []

    sorted_edges = sorted(edges, key=lambda e: e.position)
    segments = []

    for edge, next_edge in zip(sorted_edges[:-1], sorted_edges[1:]):
        start_pos = edge.position
        end_pos = next_edge.position
        distance = end_pos - start_pos

        local_bw = estimate_local_distorted_bit_width(
            mid_pos=(start_pos + end_pos) / 2,
            bit_width_px=bit_width_px,
            n_pixels=n_pixels,
            distort_coeff=distort_coeff,
        )

        if abs(distance) <= 0.5 * local_bw:
            continue

        n_bits = max(1, int(round(distance / local_bw)))
        bit_value = 1 if edge.d1_value > 0 else 0

        segments.append(BitSegment(
            start_pos=start_pos,
            end_pos=end_pos,
            bit_value=bit_value,
            n_bits=n_bits,
            distance=distance,
            measured_period=distance / n_bits,
        ))

    return segments

def build_bit_segments(
    edges: List[DetectedEdge],
    bit_width_px: float,
    n_pixels: int,
    distort_coeff: float,
    roi_start: float,
    roi_end: float
) -> List[BitSegment]:
    """
    Построение сегментов бит из фронтов.

    Логика восстановления:
    1. внутренние сегменты между соседними фронтами строятся по локальной
       оценке ширины бита;
    2. сегменты, примыкающие к границам ROI, строятся отдельно:
       - значение бита определяется по ближайшему фронту;
       - число бит считается по средней оценке периода;
       - битовая сетка якорится на ближайшем фронте, а не на границе ROI.
    """
    if not edges or roi_end <= roi_start:
        return []

    sorted_edges = sorted(edges, key=lambda e: e.position)
    segments: List[BitSegment] = []

    # 1) Сначала восстанавливаем только внутренние сегменты между фронтами
    internal_segments = build_internal_bit_segments(
        sorted_edges,
        bit_width_px=bit_width_px,
        n_pixels=n_pixels,
        distort_coeff=distort_coeff,
    )

    # 2) Средняя оценка периода нужна именно для крайних сегментов у ROI
    reference_period = compute_mean_measured_period(internal_segments)
    if reference_period <= 0:
        reference_period = bit_width_px

    # 3) Левый сегмент: roi_start -> первый фронт
    # Значение бита определяется по направлению первого фронта:
    #   D1 < 0 : 1 -> 0, значит слева был бит 1
    #   D1 > 0 : 0 -> 1, значит слева был бит 0
    first_edge = sorted_edges[0]
    left_visible_distance = first_edge.position - roi_start
    if left_visible_distance > 1e-12:
        left_bit = 1 if first_edge.d1_value < 0 else 0
        left_n_bits = estimate_roi_segment_n_bits(
            left_visible_distance,
            reference_period
        )

        if left_n_bits > 0:
            # Сетка ставится от фронта назад, а не от roi_start.
            left_grid_start = first_edge.position - left_n_bits * reference_period
            segments.append(BitSegment(
                start_pos=roi_start,
                end_pos=first_edge.position,
                bit_value=left_bit,
                n_bits=left_n_bits,
                distance=left_visible_distance,
                measured_period=reference_period,
                grid_start_pos=left_grid_start,
            ))

    # 4) Внутренние сегменты между фронтами
    segments.extend(internal_segments)

    # 5) Правый сегмент: последний фронт -> roi_end
    # Значение бита определяется по направлению последнего фронта:
    #   D1 > 0 : после фронта бит 1
    #   D1 < 0 : после фронта бит 0
    last_edge = sorted_edges[-1]
    right_visible_distance = roi_end - last_edge.position
    if right_visible_distance > 1e-12:
        right_bit = 1 if last_edge.d1_value > 0 else 0
        right_n_bits = estimate_roi_segment_n_bits(
            right_visible_distance,
            reference_period
        )

        if right_n_bits > 0:
            # Здесь сетка естественно идёт от последнего фронта вправо.
            segments.append(BitSegment(
                start_pos=last_edge.position,
                end_pos=roi_end,
                bit_value=right_bit,
                n_bits=right_n_bits,
                distance=right_visible_distance,
                measured_period=reference_period,
                grid_start_pos=last_edge.position,
            ))

    return segments


def extract_bits_with_positions(segments: List[BitSegment]) -> List[RecoveredBit]:
    """
    Извлечение бит с координатами внутри видимой части сегмента.

    Важно для крайних сегментов у ROI:
    битовая сетка может быть якорена вне ROI, поэтому начало полного бита
    может лежать за пределами видимого сегмента. В recovered_bits нужно
    добавлять все биты, которые пересекают сегмент, и ставить их позицию
    в пределах видимой области сегмента.
    """
    bits = []

    for seg_idx, seg in enumerate(segments):
        if seg.n_bits <= 0:
            continue
        if seg.measured_period <= 1e-12:
            continue
        if seg.end_pos <= seg.start_pos:
            continue

        grid_start = (
            seg.grid_start_pos
            if seg.grid_start_pos is not None
            else seg.start_pos
        )

        for i in range(seg.n_bits):
            bit_start = grid_start + i * seg.measured_period
            bit_end = bit_start + seg.measured_period

            # Видимая часть бита внутри сегмента
            visible_start = max(bit_start, seg.start_pos)
            visible_end = min(bit_end, seg.end_pos)

            # Бит добавляется, если он реально пересекает видимый сегмент
            if visible_end - visible_start <= 1e-12:
                continue

            bits.append(RecoveredBit(
                value=seg.bit_value,
                position=visible_start,
                segment_idx=seg_idx
            ))

    bits.sort(key=lambda b: b.position)
    return bits


def compute_mean_measured_period(segments: List[BitSegment]) -> float:
    """
    Средний измеренный период по всем сегментам.

    Для крайних сегментов у ROI видимая длина может быть меньше полной длины
    покрываемых ими битов, поэтому использовать sum(distance) / sum(n_bits)
    нельзя — это даст занижение периода.

    Здесь используется среднее по периодам сегментов с весом n_bits:

        mean_period = sum(measured_period_i * n_bits_i) / sum(n_bits_i)
    """
    if not segments:
        return 0.0

    total_period = sum(seg.measured_period * seg.n_bits for seg in segments)
    total_bits = sum(seg.n_bits for seg in segments)

    return total_period / total_bits if total_bits > 0 else 0.0


def detect_edges_and_recover_bits(
        adc_signal: np.ndarray,
        true_bits: np.ndarray,
        true_edges: np.ndarray,
        distort_coeff: float,
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

    bit_width_px = config.bit_width_px
    n_pixels = len(adc_signal)

    # 1. Глобальная нормализация
    signal_norm = normalize_global(adc_signal.astype(np.float64))

    # 2. Гауссово сглаживание
    signal_smoothed = gaussian_filter1d(signal_norm,
                                        sigma=config.smoothing_sigma)

    # 3. Минимаксная нормализация
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
    filtered_by_amp = [e for e in raw_edges if
                       abs(e.d1_value) >= peak_threshold]

    # 9. Фильтрация по минимальному расстоянию
    min_dist = bit_width_px * config.min_edge_distance_factor
    filtered_edges = filter_by_min_distance(filtered_by_amp, min_dist)

    # 10. ROI: из конфига, если задано; иначе — авто по крайним фронтам
    auto_roi_start = filtered_edges[0].position if filtered_edges else 0.0
    auto_roi_end = (
        filtered_edges[-1].position if filtered_edges else float(n_pixels - 1)
    )

    roi_start = (
        float(config.roi_start)
        if config.roi_start is not None
        else auto_roi_start
    )
    roi_end = (
        float(config.roi_end)
        if config.roi_end is not None
        else auto_roi_end
    )

    roi_start = float(np.clip(roi_start, 0.0, n_pixels - 1))
    roi_end = float(np.clip(roi_end, 0.0, n_pixels - 1))
    if roi_end < roi_start:
        roi_start, roi_end = roi_end, roi_start

    # Сегменты строятся только по фронтам внутри ROI
    roi_edges = [
        e for e in filtered_edges
        if roi_start <= e.position <= roi_end
    ]

    # 11. Построение сегментов бит
    bit_segments = build_bit_segments(
        roi_edges,
        bit_width_px,
        n_pixels,
        distort_coeff,
        roi_start,
        roi_end,
    )

    # 12. Измеренный средний период
    measured_bit_period = compute_mean_measured_period(bit_segments)

    # 13. Извлечение бит с координатами
    recovered_bits = extract_bits_with_positions(bit_segments)
    recovered_bit_values = np.array([b.value for b in recovered_bits])

    # Определяем, какая последовательность длиннее
    if len(true_bits) >= len(recovered_bit_values):
        longer = true_bits
        shorter = recovered_bit_values
    else:
        longer = recovered_bit_values
        shorter = true_bits

    # 14. Подсчёт ошибок: скользим shorter по longer
    window = len(shorter)
    best_errors = window + 1
    best_k = 0

    for k in range(len(longer) - window + 1):
        bit_errors = int(np.sum(shorter != longer[k: k + window]))
        if bit_errors < best_errors:
            best_errors = bit_errors
            best_k = k

    longer_aligned = longer[best_k: best_k + window]

    # 15. Точность
    total_bits = window
    correct_bits = total_bits - best_errors
    accuracy = (correct_bits / total_bits * 100) if total_bits > 0 else 0.0

    # 16. Ошибки позиционирования фронтов
    edge_errors = []
    used_detected = set()
    detected_positions = [e.position for e in roi_edges]

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
    rms_edge_error = np.sqrt(np.mean(edge_errors ** 2)) if len(
        edge_errors) > 0 else 0.0

    return EdgeDetectionResult(
        first_derivative=d1,
        second_derivative=d2,
        smoothed_signal=signal_smoothed,
        local_norm_signal=local_norm_signal,
        detected_edges=roi_edges,
        bit_width_px=bit_width_px,
        measured_bit_period=measured_bit_period,
        bit_segments=bit_segments,
        recovered_bits=recovered_bits,
        recovered_bit_values=recovered_bit_values,
        roi_start=roi_start,
        roi_end=roi_end,
        bit_errors=best_errors,
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

    distort_coeff = sim_config.distort_coeff

    br_config = BRConfig(bit_width_px=sim_config.bit_width_px)
    result = detect_edges_and_recover_bits(
        sim_result.adc_signal,
        sim_result.bits,
        sim_result.true_edges,
        distort_coeff,
        br_config
    )
    
    print(f"Detected edges: {len(result.detected_edges)}")
    print(f"Accuracy: {result.accuracy:.1f}%")
    print(f"RMS edge error: {result.rms_edge_error:.3f} px")
    print(f"True bits:      {''.join(str(b) for b in sim_result.bits)}")
    print(f"Recovered bits: {''.join(str(b) for b in result.recovered_bit_values)}")
