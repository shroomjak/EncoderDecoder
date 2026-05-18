from dataclasses import dataclass
import numpy as np


@dataclass
class DetectedEdge:
    """Совместимый тип для всех детекторов."""
    position: float
    d1_value: float  # или strength — зависит от алгоритма


@dataclass
class BitSegment:
    start_pos: float
    end_pos: float
    bit_value: int
    n_bits: int
    distance: float
    measured_period: float


@dataclass
class RecoveredBit:
    value: int
    position: float
    segment_idx: int


@dataclass
class EdgeDetectionResult:
    detected_edges: list[DetectedEdge]
    recovered_bits: list[RecoveredBit]
    recovered_bit_values: np.ndarray
    rms_edge_error: float
    accuracy: float
    edge_errors: np.ndarray
    config: object  # может быть любым Config-классом


def distort_division(pos: float, center: float, norm: float, coeff: float) -> float:
    return pos + coeff * (pos - center)**2 / norm


def build_bit_segments(
    edges: list[DetectedEdge],
    bit_width_px: float,
    n_pixels: int,
    distort_coeff: float,
    roi_end: float,
) -> list[BitSegment]:
    """
    Построение сегментов бит из фронтов.
    Используется всеми алгоритмами.
    """
    if not edges:
        return []

    sorted_edges = sorted(edges, key=lambda e: e.position)
    segments = []

    for i, edge in enumerate(sorted_edges):
        next_edge = sorted_edges[i + 1] if i + 1 < len(sorted_edges) else None
        bit_value = 1 if edge.d1_value > 0 else 0

        start_pos = edge.position
        end_pos = next_edge.position if next_edge else roi_end
        distance = end_pos - start_pos

        # Коррекция ширины бита на дисторсию
        center = n_pixels / 2
        norm = center
        pos = (start_pos + end_pos) / 2
        bit_width_px_distorted = (
            distort_division(pos + bit_width_px, center, norm, distort_coeff) -
            distort_division(pos, center, norm, distort_coeff)
        )

        n_bits = max(1, round(distance / bit_width_px_distorted))
        measured_period = distance / n_bits

        if abs(distance) > 0.5 * bit_width_px_distorted:
            segments.append(BitSegment(
                start_pos=start_pos,
                end_pos=end_pos,
                bit_value=bit_value,
                n_bits=n_bits,
                distance=distance,
                measured_period=measured_period
            ))

    return segments


def extract_bits_with_positions(segments: list[BitSegment]) -> list[RecoveredBit]:
    bits = []
    for seg_idx, seg in enumerate(segments):
        for i in range(seg.n_bits):
            position = seg.start_pos + i * seg.measured_period
            bits.append(RecoveredBit(value=seg.bit_value, position=position, segment_idx=seg_idx))
    return bits


def compute_accuracy(true_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    longer, shorter = (true_bits, recovered_bits) if len(true_bits) >= len(recovered_bits) \
                      else (recovered_bits, true_bits)
    window = len(shorter)
    best_errors = window + 1

    for k in range(len(longer) - window + 1):
        errors = int(np.sum(shorter != longer[k:k+window]))
        if errors < best_errors:
            best_errors = errors

    correct = window - best_errors
    return correct / window if window > 0 else 0.0


def compute_rms_edge_error(true_edges: np.ndarray, detected_edges: np.ndarray) -> float:
    if len(true_edges) == 0 or len(detected_edges) == 0:
        return float('inf')

    errors = []
    used = set()
    det_pos = [e.position for e in detected_edges]

    for te in true_edges:
        min_dist = float('inf')
        min_idx = -1
        for j, dp in enumerate(det_pos):
            if j in used:
                continue
            dist = abs(dp - te)
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        if min_idx >= 0 and min_dist < 10:  # допуск 10 пикселей
            errors.append(min_dist)
            used.add(min_idx)

    return np.sqrt(np.mean(np.array(errors) ** 2)) if errors else 0.0


def make_detection_result(
    detected_edges: list[DetectedEdge],
    true_bits: np.ndarray,
    true_edges: np.ndarray,
    bit_width_px: float,
    n_pixels: int,
    distort_coeff: float,
    config: object,
) -> EdgeDetectionResult:
    """
    Создает единый результат со всеми метриками.
    Вызывается из любого алгоритма.
    """
    roi_end = max(e.position for e in detected_edges) if detected_edges else 0.0
    bit_segments = build_bit_segments(
        detected_edges, bit_width_px, n_pixels, distort_coeff, roi_end
    )
    recovered_bits = extract_bits_with_positions(bit_segments)
    recovered_bit_values = np.array([b.value for b in recovered_bits])

    accuracy = compute_accuracy(true_bits, recovered_bit_values)
    edge_errors = []  # можно добавить, если нужно визуализировать
    rms_edge_error = compute_rms_edge_error(true_edges, detected_edges)

    return EdgeDetectionResult(
        detected_edges=detected_edges,
        recovered_bits=recovered_bits,
        recovered_bit_values=recovered_bit_values,
        rms_edge_error=rms_edge_error,
        accuracy=accuracy,
        edge_errors=np.array(edge_errors),
        config=config,
    )