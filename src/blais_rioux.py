"""
blais_rioux.py — Blais-Rioux edge detector + bit recovery.

Pipeline overview
-----------------
detect_edges(adc_signal, config):
    0. Undistort signal via division model (if config.distort_coeff != 0).
       Geometry-correct: N pixels -> M pixels, same 1-px scale, bicubic spline.
       undistorted_signal stored in EdgeDetectionResult has length M.
    1. Global normalisation
    2. Gaussian smoothing
    3. Local min-max normalisation (sliding window)
    4. D1 -- BR moment filter
    5. D2 = D1(D1)
    6. Zero-crossings of D2 -> candidate edges
    7. Amplitude threshold on |D1|
    8. Min-distance filter
    -> EdgeDetectionResult (all positions in undistorted M-point index space)

recover_bits(edge_result, true_bits, true_edges):
    Works entirely in undistorted M-point coordinates.
    bit_width_px is interpreted in the new (expanded) pixel scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, correlate1d


VIGNETTE = np.array([
        811.755980861244,878.569377990431,983.239234449761,1088.28229665072,
        1196.40669856459,1291.25358851675,1383.07177033493,1477.44976076555,
        1568.28229665072,1694.36363636364,1780.23444976077,1836.80861244019,
        1933.66028708134,2032.55502392344,2117.20574162679,2211.68421052632,
        2335.23444976077,2480.13397129187,2485.52153110048,2674.27751196172,
        2699.29186602871,2796.04784688995,2808.95215311005,2889.05263157895,
        2942.07655502392,3077.63636363636,3113.16746411483,3217.28708133971,
        3264.55502392345,3314.30622009569,3378.33014354067,3563.14354066986,
        3392.9043062201,3615.13397129187,3593.30622009569,3613.94258373206,
        3779.3014354067,3773.54545454545,3822.08612440191,3869.44019138756,
        3928.43062200957,4028.53588516746,3859.67942583732,3726.91866028708,
        3627.57894736842,3537.01913875598,3425.47368421053,3385.63636363636,
        3371.51674641148,3327.95215311005,3286.21531100479,3273.53588516746,
        3219.12440191388,3201.99043062201,3166.33971291866,3158.02870813397,
        3030.7990430622,3080.71291866029,3087.54066985646,3092.36363636364,
        3112.03827751196,3131.14832535885,3069.49282296651,3019.66985645933,
        3158.87559808612,3217.53588516746,3182.51674641148,3241.75119617225,
        3165.0956937799,3183.65071770335,3147.15789473684,3157.17224880383,
        3107.97129186603,3084.54545454545,3056.37799043062,3064.67942583732,
        2988.66985645933,3023.48803827751,2989.89473684211,2929.58373205742,
        2956.58851674641,2896.17703349282,2872.22009569378,2861.61244019139,
        2862.90909090909,2831.05741626794,2799.88038277512,2781.11961722488,
        2670.96172248804,2639.2009569378,2629.74641148325,2558.95693779904,
        2525.83732057416,2447.47846889952,2366.4976076555,2311.54066985646,
        2240.41626794258,2071.95215311005,1902.68899521531,1833.71770334928,
        1768.63157894737,1729.995215311,1604.61722488038,1495.90909090909,
        1407.67942583732,1334.91866028708,1243.22488038278,1178.88995215311
    ],
    dtype=np.float32
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _sensor_center_scale(n: int):
    c = 0.5 * (n - 1)
    return c, max(c, 1.0)

def _undistort_and_resample(
    signal: np.ndarray,
    k: float,
    output_step_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Geometry-correct 1-D undistortion (division model).

    Algorithm
    ---------
    1. Source uniform grid x_d = 0..N-1 (distorted sensor).
    2. Inverse transform: x_u = undistort(x_d) -- non-uniform.
    3. New uniform grid x_u_new with step output_step_px,
       spanning [min(x_u), max(x_u)]. Length M >= N.
    4. Natural cubic spline interpolation: signal(x_u) -> x_u_new.

    Returns
    -------
    x_u_new : ndarray (M,)  -- new axis in pixel units (step = output_step_px)
    undist  : ndarray (M,)  -- undistorted signal values
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < 2 or abs(k) < 1e-15:
        return np.arange(n, dtype=np.float64), signal.copy()

    if output_step_px <= 0:
        raise ValueError("output_step_px must be > 0")

    c, s = _sensor_center_scale(n)
    x_d_norm = (np.arange(n, dtype=np.float64) - c) / s

    # Inverse division model: x_u = 2*x_d / (1 + sqrt(1 - 4*k*x_d^2))
    disc    = np.clip(1.0 - 4.0 * k * x_d_norm ** 2, 0.0, None)
    x_u_norm = 2.0 * x_d_norm / (1.0 + np.sqrt(disc))
    x_u_px   = x_u_norm * s + c  # non-uniform grid in undistorted space

    # New uniform grid spanning the same range
    x_min, x_max = float(x_u_px[0]), float(x_u_px[-1])
    m = int(np.floor((x_max - x_min) / output_step_px)) + 1
    x_u_new = np.arange(m, dtype=np.float64) * output_step_px

    # Natural cubic spline interpolation
    x_u_px_shifted = x_u_px - x_min  # нормируем к [0, x_max - x_min]
    spline = CubicSpline(x_u_px_shifted, signal, bc_type="natural")
    undist = spline(x_u_new)

    return x_u_new, undist

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BRConfig:
    """Blais-Rioux detector configuration."""

    filter_order: int = 2
    """|KIR kernel order N (kernel length = 2N+1): 2 or 4."""

    peak_threshold_rel: float = 0.25
    """|D1| threshold relative to maximum (0..1)."""

    min_edge_distance_factor: float = 0.3
    """Minimum inter-edge distance as a fraction of bit_width_px."""

    bit_width_px: float = 5
    """Prior bit width in undistorted pixels."""

    smoothing_sigma: float = 0.2
    """Gaussian smoothing sigma before differentiation."""

    distort_coeff: float = 0.0
    """Division-model coefficient k. 0 = no correction.
    When non-zero, the signal is geometry-corrected (N->M, bicubic spline)
    before all further processing. All subsequent coordinates are M-point."""

    undist_output_step_px: float = 1.0
    """Output grid step for undistortion (normally 1.0 -- one pixel)."""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectedEdge:
    position:  float   # sub-pixel index in undistorted M-point coordinates
    d1_value:  float   # D1 at this position (sign = transition direction)

@dataclass
class BitSegment:
    start_pos:       float
    end_pos:         float
    bit_value:       int
    n_bits:          int
    distance:        float
    measured_period: float
    grid_start_pos:  Optional[float] = None

@dataclass
class RecoveredBit:
    value:       int
    position:    float   # undistorted pixel index
    segment_idx: int

@dataclass
class EdgeDetectionResult:
    """Output of detect_edges().

    raw_signal         -- original ADC values,         length N
    undistorted_signal -- after bicubic undistortion,  length M  (M >= N)
    undist_x           -- uniform x-axis for undistorted_signal (pixel units)

    All edge / bit positions are indices 0..M-1 into undistorted_signal.
    """
    raw_signal:          np.ndarray   # float64, shape (N,)
    undistorted_signal:  np.ndarray   # float64, shape (M,)
    undist_x:            np.ndarray   # float64, shape (M,)

    first_derivative:    np.ndarray
    second_derivative:   np.ndarray
    smoothed_signal:     np.ndarray
    vignette_norm:   np.ndarray

    detected_edges:  List[DetectedEdge]
    bit_width_px:    float
    peak_threshold:  float
    config:          BRConfig = field(repr=False)

@dataclass
class BitRecoveryResult:
    """Output of recover_bits()."""
    edge_result: EdgeDetectionResult

    measured_bit_period: float
    bit_segments:        List[BitSegment]
    recovered_bits:      List[RecoveredBit]
    recovered_bit_values: np.ndarray

    bit_errors:    int
    edge_errors:   np.ndarray
    rms_edge_error: float
    accuracy:      float

    # Convenience forwarding from EdgeDetectionResult
    @property
    def raw_signal(self):           return self.edge_result.raw_signal
    @property
    def undistorted_signal(self):   return self.edge_result.undistorted_signal
    @property
    def undist_x(self):             return self.edge_result.undist_x
    @property
    def first_derivative(self):     return self.edge_result.first_derivative
    @property
    def second_derivative(self):    return self.edge_result.second_derivative
    @property
    def smoothed_signal(self):      return self.edge_result.smoothed_signal
    @property
    def local_norm_signal(self):    return self.edge_result.local_norm_signal
    @property
    def detected_edges(self):       return self.edge_result.detected_edges
    @property
    def bit_width_px(self):         return self.edge_result.bit_width_px
    @property
    def peak_threshold(self):       return self.edge_result.peak_threshold
    @property
    def config(self):               return self.edge_result.config

# ---------------------------------------------------------------------------
# DSP primitives
# ---------------------------------------------------------------------------

def make_br_kernel(order: int) -> np.ndarray:
    k = np.array([-1] * order + [0] + [1] * order, dtype=np.float64)
    return k / np.sum(np.abs(k))

def correlate_mirror(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return correlate1d(signal, kernel, mode="reflect")

def normalize_global(signal: np.ndarray) -> np.ndarray:
    s_min, s_max = signal.min(), signal.max()
    return (signal - s_min) / (s_max - s_min + 1e-12)

def apply_minmax_normalization(signal: np.ndarray, window_size: int) -> np.ndarray:
    global_min  = float(signal.min())
    running_max = maximum_filter1d(signal, size=window_size, mode="reflect")
    local_max   = np.maximum(running_max, 0.5)
    return (signal - global_min) / (local_max - global_min + 1e-12)

def find_zero_crossings_d2(d2: np.ndarray, d1: np.ndarray) -> List[DetectedEdge]:
    """Sub-pixel zero-crossings of D2. Positions are indices 0..M-1."""
    signs = np.sign(d2)
    cross = np.where((signs[:-1] * signs[1:]) < 0)[0]
    edges = []
    for i in cross:
        pos  = i + d2[i] / (d2[i] - d2[i + 1])
        frac = pos - i
        d1v  = d1[i] * (1 - frac) + d1[i + 1] * frac
        edges.append(DetectedEdge(position=pos, d1_value=d1v))
    return edges

def filter_by_min_distance(
    edges: List[DetectedEdge], min_dist: float
) -> List[DetectedEdge]:
    if not edges:
        return []
    result = [sorted(edges, key=lambda e: e.position)[0]]
    for curr in sorted(edges, key=lambda e: e.position)[1:]:
        if curr.position - result[-1].position < min_dist:
            if abs(curr.d1_value) > abs(result[-1].d1_value):
                result[-1] = curr
        else:
            result.append(curr)
    return result

# ---------------------------------------------------------------------------
# Bit-segment helpers (all in undistorted M-point index space)
# ---------------------------------------------------------------------------

def _estimate_segment_n_bits(visible_distance: float, reference_period: float) -> int:
    if visible_distance <= 1e-12:
        return 0
    return max(1, int(np.ceil((visible_distance - 1e-9) / max(reference_period, 1e-12))))

def _build_internal_segments(
    edges: List[DetectedEdge],
    bit_width_px: float,
) -> List[BitSegment]:
    segs = []
    for e, ne in zip(edges[:-1], edges[1:]):
        d = ne.position - e.position
        if d <= 0.5 * bit_width_px:
            continue
        n = max(1, int(round(d / bit_width_px)))
        segs.append(BitSegment(
            start_pos=e.position,
            end_pos=ne.position,
            bit_value=1 if e.d1_value > 0 else 0,
            n_bits=n,
            distance=d,
            measured_period=d / n,
        ))
    return segs

def _compute_mean_period(segments: List[BitSegment]) -> float:
    total_p = sum(s.measured_period * s.n_bits for s in segments)
    total_n = sum(s.n_bits for s in segments)
    return total_p / total_n if total_n > 0 else 0.0

def build_bit_segments(
    edges: List[DetectedEdge],
    bit_width_px: float,
    signal_start: float,
    signal_end: float,
) -> List[BitSegment]:
    if not edges or signal_end <= signal_start:
        return []

    sorted_edges = sorted(edges, key=lambda e: e.position)
    internal     = _build_internal_segments(sorted_edges, bit_width_px)
    ref_period   = _compute_mean_period(internal)
    if ref_period <= 0:
        ref_period = bit_width_px

    segs: List[BitSegment] = []

    fe     = sorted_edges[0]
    left_d = fe.position - signal_start
    if left_d > 1e-12:
        n = _estimate_segment_n_bits(left_d, ref_period)
        if n > 0:
            segs.append(BitSegment(
                start_pos=signal_start,
                end_pos=fe.position,
                bit_value=1 if fe.d1_value < 0 else 0,
                n_bits=n,
                distance=left_d,
                measured_period=ref_period,
                grid_start_pos=fe.position - n * ref_period,
            ))

    segs.extend(internal)

    le      = sorted_edges[-1]
    right_d = signal_end - le.position
    if right_d > 1e-12:
        n = _estimate_segment_n_bits(right_d, ref_period)
        if n > 0:
            segs.append(BitSegment(
                start_pos=le.position,
                end_pos=signal_end,
                bit_value=1 if le.d1_value > 0 else 0,
                n_bits=n,
                distance=right_d,
                measured_period=ref_period,
                grid_start_pos=le.position,
            ))

    return segs

def extract_bits_with_positions(segments: List[BitSegment]) -> List[RecoveredBit]:
    bits = []
    for idx, seg in enumerate(segments):
        if seg.n_bits <= 0 or seg.measured_period <= 1e-12:
            continue
        gs  = seg.grid_start_pos if seg.grid_start_pos is not None else seg.start_pos
        eps = max(1e-9, 1e-6 * seg.measured_period)
        for i in range(seg.n_bits):
            bs = gs + i * seg.measured_period
            be = bs + seg.measured_period
            if bs >= seg.start_pos - eps and be <= seg.end_pos + eps:
                bits.append(RecoveredBit(value=seg.bit_value, position=bs, segment_idx=idx))
    bits.sort(key=lambda b: b.position)
    return bits

def compute_mean_measured_period(segments: List[BitSegment]) -> float:
    return _compute_mean_period(segments)

# ---------------------------------------------------------------------------
# Sliding-window bit accuracy
# ---------------------------------------------------------------------------

def _sliding_window_accuracy(
    rec_values: np.ndarray,
    true_bits: np.ndarray,
) -> tuple[int, float]:
    """
    Find the best-matching position of rec_values inside true_bits
    by sliding rec_values over true_bits (or vice versa if rec is longer).

    Returns
    -------
    best_err : int    -- number of mismatched bits at best offset
    accuracy : float  -- percentage of correct bits (0..100)
    """
    n_rec  = len(rec_values)
    n_true = len(true_bits)

    if n_rec == 0 or n_true == 0:
        return 0, 0.0

    rec  = rec_values.astype(np.int8)
    true = true_bits.astype(np.int8)

    if n_rec <= n_true:
        # Slide rec over true: for each offset k, count matching bits
        win = n_rec
        n_offsets = n_true - win + 1
        best_matches = 0
        for k in range(n_offsets):
            matches = int(np.sum(rec == true[k:k + win]))
            if matches > best_matches:
                best_matches = matches
        best_err = win - best_matches
        accuracy = best_matches / win * 100.0
    else:
        # rec is longer than true: slide true over rec
        win = n_true
        n_offsets = n_rec - win + 1
        best_matches = 0
        for k in range(n_offsets):
            matches = int(np.sum(true == rec[k:k + win]))
            if matches > best_matches:
                best_matches = matches
        best_err = win - best_matches
        # accuracy denominator = n_rec (we recovered more bits than ground truth)
        accuracy = best_matches / n_rec * 100.0

    return best_err, accuracy

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_edges(
    adc_signal: np.ndarray,
    config: Optional[BRConfig] = None,
) -> EdgeDetectionResult:
    """
    Steps 0-8: undistort (N->M, bicubic) -> normalise -> D1/D2 -> edges.

    Step 0 -- geometry-correct undistortion:
        x_d grid (N px, distorted) -> x_u_new grid (M px, same 1-px scale).
        M > N for pincushion distortion (k > 0).
        Interpolation: natural cubic spline.
        k == 0 => undistorted_signal == raw_signal, M == N.

    All subsequent positions (edges, bits) are indices 0..M-1.
    No ROI clipping — full signal width is processed.
    """
    if config is None:
        config = BRConfig()

    raw = adc_signal.astype(np.float64)

    # Step 0 -- geometry-correct undistortion (N -> M)
    x_u, undist = _undistort_and_resample(
        raw, config.distort_coeff, config.undist_output_step_px
    )

    _, undist_vignette = _undistort_and_resample(
        VIGNETTE, config.distort_coeff, config.undist_output_step_px
    )
    m = len(undist)

    # Steps 1-3
    sig_norm   = normalize_global(undist)
    sig_smooth = gaussian_filter1d(sig_norm, sigma=config.smoothing_sigma)
    vignette_norm = normalize_global(sig_smooth / undist_vignette)
    # Steps 4-5
    kernel = make_br_kernel(config.filter_order)
    d1     = correlate_mirror(vignette_norm, kernel)
    d2     = correlate_mirror(d1, kernel)

    # Step 6
    raw_edges = find_zero_crossings_d2(d2, d1)

    # Steps 7-8
    max_amp  = max((abs(e.d1_value) for e in raw_edges), default=0.0)
    peak_thr = max_amp * config.peak_threshold_rel
    edges    = filter_by_min_distance(
        [e for e in raw_edges if abs(e.d1_value) >= peak_thr],
        config.bit_width_px * config.min_edge_distance_factor,
    )

    return EdgeDetectionResult(
        raw_signal=raw,
        undistorted_signal=undist,
        undist_x=x_u,
        first_derivative=d1,
        second_derivative=d2,
        smoothed_signal=sig_smooth,
        vignette_norm=vignette_norm,
        detected_edges=edges,
        bit_width_px=config.bit_width_px,
        peak_threshold=peak_thr,
        config=config,
    )


def recover_bits(
    edge_result: EdgeDetectionResult,
    true_bits:   np.ndarray,
    true_edges:  np.ndarray,
) -> BitRecoveryResult:
    """
    Steps 10-15: segments -> bits -> metrics.

    Operates entirely in undistorted M-point index space.
    true_edges must be in the same (undistorted) coordinates.
    Processes the full signal width — no ROI clipping.
    """
    edges  = edge_result.detected_edges
    bw     = edge_result.bit_width_px
    m      = len(edge_result.undistorted_signal)

    # Full signal range (no ROI)
    signal_start = 0.0
    signal_end   = float(m - 1)

    # Step 10
    segments = build_bit_segments(edges, bw, signal_start, signal_end)

    # Step 11
    mean_period = _compute_mean_period(segments)

    # Step 12
    rec_bits   = extract_bits_with_positions(segments)
    rec_values = np.array([b.value for b in rec_bits])

    # Steps 13-14 -- bit accuracy via corrected sliding window match
    best_err, accuracy = _sliding_window_accuracy(rec_values, true_bits)

    # Step 15 -- edge position errors (in undistorted px)
    det_pos    = [e.position for e in edges]
    errs, used = [], set()
    for te in true_edges:
        best_d, best_j = float("inf"), -1
        for j, dp in enumerate(det_pos):
            if j not in used and abs(dp - te) < best_d:
                best_d, best_j = abs(dp - te), j
        if best_j >= 0 and best_d < bw * 0.5:
            errs.append(det_pos[best_j] - te)
            used.add(best_j)

    errs_arr = np.array(errs)
    rms      = float(np.sqrt(np.mean(errs_arr ** 2))) if len(errs_arr) > 0 else 0.0

    return BitRecoveryResult(
        edge_result=edge_result,
        measured_bit_period=mean_period,
        bit_segments=segments,
        recovered_bits=rec_bits,
        recovered_bit_values=rec_values,
        bit_errors=best_err,
        edge_errors=errs_arr,
        rms_edge_error=rms,
        accuracy=accuracy,
    )


def detect_edges_and_recover_bits(
    adc_signal: np.ndarray,
    true_bits:  np.ndarray,
    true_edges: np.ndarray,
    config: Optional[BRConfig] = None,
) -> BitRecoveryResult:
    er = detect_edges(adc_signal, config)
    return recover_bits(er, true_bits, true_edges)