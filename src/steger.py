from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .common_processing import make_detection_result, DetectedEdge, EdgeDetectionResult


@dataclass
class StegerConfig:
    sigma: float = 2.0
    magnitude_thresh: float = 0.15
    min_distance_px: float = 2.0
    bit_width_px: float = 8.0


def _gaussian_derivative(signal: np.ndarray, sigma: float, order: int) -> np.ndarray:
    return gaussian_filter1d(signal, sigma=sigma, order=order, mode='reflect')


def _find_zero_crossings_subpixel(
    g1: np.ndarray,
    g2: np.ndarray,
    g3: np.ndarray,
    mag_thresh_abs: float,
) -> list[DetectedEdge]:
    positions = []
    strengths = []

    for i in range(1, len(g2) - 1):
        if g2[i - 1] * g2[i + 1] >= 0:
            continue
        if abs(g1[i]) < mag_thresh_abs:
            continue

        offset = -g2[i] / g3[i] if abs(g3[i]) > 1e-12 else 0
        offset = np.clip(offset, -1.0, 1.0)
        x_sub = i + offset

        if 0 <= x_sub < len(g2):
            positions.append(x_sub)
            strengths.append(abs(g1[i]))

    # Подавление близких
    combined = sorted(zip(positions, strengths), key=lambda x: -x[1])
    kept = []
    for pos, strn in combined:
        if all(abs(pos - p) >= 2.0 for p in kept):
            kept.append(pos)

    return [DetectedEdge(position=p, d1_value=0.0) for p in kept]


def detect_edges_steger(
    adc_signal: np.ndarray,
    true_bits: np.ndarray,
    true_edges: np.ndarray,
    distort_coeff: float,
    config: StegerConfig = None,
) -> EdgeDetectionResult:
    if config is None:
        config = StegerConfig()

    signal = adc_signal.astype(float)
    g1 = _gaussian_derivative(signal, config.sigma, 1)
    g2 = _gaussian_derivative(signal, config.sigma, 2)
    g3 = _gaussian_derivative(signal, config.sigma, 3)

    max_g1 = np.max(np.abs(g1))
    mag_thresh_abs = config.magnitude_thresh * max_g1

    detected_edges = _find_zero_crossings_subpixel(g1, g2, g3, mag_thresh_abs)

    # Присвоим направление: оценим знак градиента слева и справа
    for edge in detected_edges:
        pos = int(round(edge.position))
        if 0 < pos < len(signal) - 1:
            grad = signal[pos + 1] - signal[pos - 1]
            edge.d1_value = grad

    return make_detection_result(
        detected_edges=detected_edges,
        true_bits=true_bits,
        true_edges=true_edges,
        bit_width_px=config.bit_width_px,
        n_pixels=len(adc_signal),
        distort_coeff=distort_coeff,
        config=config,
    )