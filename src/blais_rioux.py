"""
blais_rioux.py — Blais-Rioux edge detector + bit recovery.

Pipeline overview
-----------------
detect_edges(adc_signal, config, vignette):
  0. Undistort signal via division model (if config.distort_coeff != 0).
     Geometry-correct: N pixels -> M pixels, same 1-px scale, bicubic spline.
     undistorted_signal stored in EdgeDetectionResult has length M.
  1. Global normalisation
  2. Gaussian smoothing
  3. Vignette correction + local min-max normalisation
  4. D1 -- BR moment filter
  5. D2 = D1(D1)
  6. Zero-crossings of D2 -> candidate edges
  7. Amplitude threshold on |D1|
  8. Min-distance filter
  -> EdgeDetectionResult (all positions in undistorted M-point index space)

recover_bits(edge_result, true_bits, true_edges):
  Works entirely in undistorted M-point coordinates.
  bit_width_px is interpreted in the new (expanded) pixel scale.

Distortion coefficients and vignette curves are stored per sensor head:
  DISTORT_K_1, DISTORT_K_2  — division-model k for heads 1 and 2
  VIGNETTE_1,  VIGNETTE_2   — flat-field reference curves for heads 1 and 2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, correlate1d

# ---------------------------------------------------------------------------
# Per-head calibration data
# ---------------------------------------------------------------------------

VIGNETTE_1 = np.array([
    1289.38228941685,1355.686825054,1447.25269978402,1544.08207343413,
    1603.656587473,1660.54427645788,1711.84665226782,1824.21166306695,
    1919.57235421166,2047.32181425486,2149.27429805616,2211.58963282937,
    2328.71706263499,2450.60043196544,2605.29589632829,2673.50539956803,
    2769.96976241901,2980.21166306695,2957.90280777538,3209.31101511879,
    3195.4060475162,3264.33477321814,3301.22894168467,3430.39740820734,
    3448.15334773218,3675.95032397408,3671.64578833693,3746.52915766739,
    3802.39740820734,3972.19654427646,3949.76457883369,4091.64794816415,
    3850.52051835853,4078.01943844492,4085.76241900648,4093.40820734341,
    4094.02591792657,4095,4094.83585313175,4094.86393088553,4094.63066954644,
    4094.66306695464,4095,4094.46004319654,4093.96112311015,4081.97624190065,
    3995.25701943845,3972.81641468683,3895.48164146868,3819.66954643629,
    3773.88552915767,3765.68466522678,3663.71490280778,3651.09071274298,
    3559.13390928726,3474.70194384449,3308.20518358531,3346.61339092873,
    3326.97192224622,3314.12095032397,3330.14902807775,3320.91144708423,
    3254.86393088553,3258.19222462203,3452.58747300216,3526.67386609071,
    3472.31533477322,3501.94384449244,3436.73650107991,3488.11447084233,
    3449.32181425486,3497.35421166307,3379.67818574514,3390.47300215983,
    3348.65226781857,3307.68034557235,3204.91792656588,3280.5161987041,
    3269.98920086393,3246.43628509719,3290.83369330454,3222.47084233261,
    3207.49676025918,3318.12742980562,3221.49028077754,3127.94600431965,
    3087.42548596112,3021.0777537797,2821.10583153348,2730.2807775378,
    2769.25917926566,2696.17494600432,2582.43628509719,2514.65226781857,
    2441.69330453564,2441.36069114471,2230.15118790497,2064.686825054,
    1957.98056155508,1894.56803455724,1825.64794816415,1827.29805615551,
    1718.2181425486,1615.02591792657,1586.20734341253,1484.14902807775,
    1402.23974082073,1333.656587473
], dtype=np.float32)

VIGNETTE_2 = np.array([
    1186.73265306122,1224.31632653061,1318.71224489796,1342.35510204082,
    1474.97755102041,1562.52244897959,1677.43469387755,1755.54081632653,
    1841.71836734694,1872.72448979592,1984.52244897959,2102.15714285714,
    2122.87346938776,2199.42040816327,2312.99387755102,2380.33265306122,
    2462.34081632653,2512.55306122449,2494.16326530612,2672.26734693878,
    2765.46734693878,2756.86326530612,2800.43469387755,2844.3306122449,
    2906.44285714286,2970.75918367347,2794.73469387755,3049.77959183673,
    3067.3693877551,3042.03265306122,3101.03673469388,3130.58571428571,
    3166.41020408163,3217.04693877551,3336.00408163265,3219.35510204082,
    3278.10612244898,3313.0693877551,3371.80816326531,3429.53469387755,
    3532.12857142857,3590.1,3650.31020408163,3668.25510204082,
    3882.38979591837,3818.22448979592,3954.61224489796,4005.33673469388,
    3985.26734693878,4042.15510204082,4008.73469387755,3874.95714285714,
    4010.94897959184,4018.37551020408,3839.49183673469,3878.99387755102,
    3750.10204081633,3826.75714285714,3805.12040816327,3747.5306122449,
    3754.75102040816,3730.94693877551,3701.53469387755,3650.70816326531,
    3653.24897959184,3537.57551020408,3532.76734693878,3535.71224489796,
    3579.26326530612,3592.32448979592,3568.40408163265,3531.6612244898,
    3461.18775510204,3594.51428571429,3496.37142857143,3487.79795918367,
    3510.54285714286,3419.75510204082,3433.20204081633,3416.46734693878,
    3409.21632653061,3327.66530612245,3304.47959183673,3190.41428571429,
    3176.45918367347,3126.71020408163,2828.4387755102,2964.55510204082,
    2969.84285714286,2881.71428571429,2793.8,2741.29183673469,2677.5306122449,
    2609.51224489796,2552.8,2428.29387755102,2358.84489795918,2249.99591836735,
    2166.33469387755,2075.71632653061,2007.01020408163,1900.98367346939,
    1824.26326530612,1746.94285714286,1697.22448979592,1590.9693877551,
    1549.08163265306,1489.91632653061
], dtype=np.float32)

DISTORT_K_1: float = 0.052
DISTORT_K_2: float = 0.050

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

    Returns
    -------
    x_u_new : ndarray (M,)
    undist  : ndarray (M,)
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < 2 or abs(k) < 1e-15:
        return np.arange(n, dtype=np.float64), signal.copy()

    if output_step_px <= 0:
        raise ValueError("output_step_px must be > 0")

    c, s = _sensor_center_scale(n)
    x_d_norm = (np.arange(n, dtype=np.float64) - c) / s

    disc = np.clip(1.0 - 4.0 * k * x_d_norm ** 2, 0.0, None)
    x_u_norm = 2.0 * x_d_norm / (1.0 + np.sqrt(disc))
    x_u_px = x_u_norm * s + c

    x_min = float(x_u_px[0])
    x_max = float(x_u_px[-1])
    m = int(np.floor((x_max - x_min) / output_step_px)) + 1
    x_u_new = np.arange(m, dtype=np.float64) * output_step_px

    spline = CubicSpline(x_u_px - x_min, signal, bc_type="natural")
    return x_u_new, spline(x_u_new)


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
    """Division-model coefficient k. 0 = no correction."""

    undist_output_step_px: float = 1.0
    """Output grid step for undistortion (normally 1.0 -- one pixel)."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectedEdge:
    position: float
    d1_value: float


@dataclass
class BitSegment:
    start_pos: float
    end_pos: float
    bit_value: int
    n_bits: int
    distance: float
    measured_period: float
    grid_start_pos: Optional[float] = None


@dataclass
class RecoveredBit:
    value: int
    position: float
    segment_idx: int


@dataclass
class EdgeDetectionResult:
    """Output of detect_edges()."""
    raw_signal: np.ndarray
    undistorted_signal: np.ndarray
    undist_x: np.ndarray

    first_derivative: np.ndarray
    second_derivative: np.ndarray
    smoothed_signal: np.ndarray
    vignette_norm: np.ndarray

    detected_edges: List[DetectedEdge]
    bit_width_px: float
    peak_threshold: float
    config: BRConfig = field(repr=False)


@dataclass
class BitRecoveryResult:
    """Output of recover_bits()."""
    edge_result: EdgeDetectionResult

    measured_bit_period: float
    bit_segments: List[BitSegment]
    recovered_bits: List[RecoveredBit]
    recovered_bit_values: np.ndarray

    bit_errors: int
    edge_errors: np.ndarray
    rms_edge_error: float
    accuracy: float

    @property
    def raw_signal(self): return self.edge_result.raw_signal
    @property
    def undistorted_signal(self): return self.edge_result.undistorted_signal
    @property
    def undist_x(self): return self.edge_result.undist_x
    @property
    def first_derivative(self): return self.edge_result.first_derivative
    @property
    def second_derivative(self): return self.edge_result.second_derivative
    @property
    def smoothed_signal(self): return self.edge_result.smoothed_signal
    @property
    def detected_edges(self): return self.edge_result.detected_edges
    @property
    def bit_width_px(self): return self.edge_result.bit_width_px
    @property
    def peak_threshold(self): return self.edge_result.peak_threshold
    @property
    def config(self): return self.edge_result.config


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
    global_min = float(signal.min())
    running_max = maximum_filter1d(signal, size=window_size, mode="reflect")
    local_max = np.maximum(running_max, 0.5)
    return (signal - global_min) / (local_max - global_min + 1e-12)


def find_zero_crossings_d2(d2: np.ndarray, d1: np.ndarray) -> List[DetectedEdge]:
    """Sub-pixel zero-crossings of D2 — fully vectorised."""
    signs = np.sign(d2)
    idx = np.where(signs[:-1] * signs[1:] < 0)[0]
    if idx.size == 0:
        return []

    d2_i  = d2[idx]
    d2_i1 = d2[idx + 1]
    frac   = d2_i / (d2_i - d2_i1)
    pos    = idx + frac
    d1_val = d1[idx] * (1.0 - frac) + d1[idx + 1] * frac

    return [DetectedEdge(position=float(p), d1_value=float(v))
            for p, v in zip(pos, d1_val)]


def filter_by_min_distance(
    edges: List[DetectedEdge], min_dist: float
) -> List[DetectedEdge]:
    if not edges:
        return []

    sorted_edges = sorted(edges, key=lambda e: e.position)
    result = [sorted_edges[0]]

    for curr in sorted_edges[1:]:
        prev = result[-1]
        if curr.position - prev.position < min_dist:
            if abs(curr.d1_value) > abs(prev.d1_value):
                result[-1] = curr
        else:
            result.append(curr)

    return result


# ---------------------------------------------------------------------------
# Bit-segment helpers
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
    half_bw = 0.5 * bit_width_px
    for e, ne in zip(edges[:-1], edges[1:]):
        d = ne.position - e.position
        if d <= half_bw:
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
    if not segments:
        return 0.0
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
    internal   = _build_internal_segments(sorted_edges, bit_width_px)
    ref_period = _compute_mean_period(internal) or bit_width_px

    segs: List[BitSegment] = []

    fe     = sorted_edges[0]
    left_d = fe.position - signal_start
    if left_d > 1e-12:
        n = _estimate_segment_n_bits(left_d, ref_period)
        if n > 0:
            segs.append(BitSegment(
                start_pos=signal_start, end_pos=fe.position,
                bit_value=1 if fe.d1_value < 0 else 0,
                n_bits=n, distance=left_d, measured_period=ref_period,
                grid_start_pos=fe.position - n * ref_period,
            ))

    segs.extend(internal)

    le      = sorted_edges[-1]
    right_d = signal_end - le.position
    if right_d > 1e-12:
        n = _estimate_segment_n_bits(right_d, ref_period)
        if n > 0:
            segs.append(BitSegment(
                start_pos=le.position, end_pos=signal_end,
                bit_value=1 if le.d1_value > 0 else 0,
                n_bits=n, distance=right_d, measured_period=ref_period,
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
        p   = seg.measured_period
        for i in range(seg.n_bits):
            bs = gs + i * p
            be = bs + p
            if bs >= seg.start_pos - eps and be <= seg.end_pos + eps:
                bits.append(RecoveredBit(value=seg.bit_value,
                                         position=bs,
                                         segment_idx=idx))
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
    n_rec  = len(rec_values)
    n_true = len(true_bits)

    if n_rec == 0 or n_true == 0:
        return 0, 0.0

    rec  = rec_values.astype(np.int8)
    true = true_bits.astype(np.int8)

    if n_rec <= n_true:
        win      = n_rec
        windows  = np.lib.stride_tricks.sliding_window_view(true, win)
        matches  = int(np.sum(windows == rec, axis=1).max())
        accuracy = matches / win * 100.0
        return win - matches, accuracy
    else:
        win      = n_true
        windows  = np.lib.stride_tricks.sliding_window_view(rec, win)
        matches  = int(np.sum(windows == true, axis=1).max())
        accuracy = matches / n_rec * 100.0
        return win - matches, accuracy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_edges(
    adc_signal: np.ndarray,
    config: Optional[BRConfig] = None,
    vignette: Optional[np.ndarray] = None,
) -> EdgeDetectionResult:
    """
    Steps 0-8: undistort -> vignette correction -> D1/D2 -> edges.

    Parameters
    ----------
    adc_signal : raw ADC samples
    config     : BRConfig (default constructed if None)
    vignette   : flat-field reference array for this head.
                 If None, falls back to VIGNETTE_1 for backward compatibility.
    """
    if config is None:
        config = BRConfig()
    if vignette is None:
        vignette = VIGNETTE_1  # backward-compat default

    raw = adc_signal.astype(np.float64)

    # Step 0 — geometry-correct undistortion (N -> M)
    x_u, undist = _undistort_and_resample(
        raw, config.distort_coeff, config.undist_output_step_px
    )
    _, undist_vignette = _undistort_and_resample(
        np.asarray(vignette, dtype=np.float64),
        config.distort_coeff,
        config.undist_output_step_px,
    )

    # Steps 1-3
    sig_norm   = normalize_global(undist)
    sig_smooth = gaussian_filter1d(sig_norm, sigma=config.smoothing_sigma)
    vignette_norm = normalize_global(sig_smooth / (undist_vignette + 1e-12))

    # Steps 4-5
    kernel = make_br_kernel(config.filter_order)
    d1 = correlate_mirror(vignette_norm, kernel)
    d2 = correlate_mirror(d1, kernel)

    # Step 6
    raw_edges = find_zero_crossings_d2(d2, d1)

    # Steps 7-8
    max_amp  = max((abs(e.d1_value) for e in raw_edges), default=0.0)
    peak_thr = max_amp * config.peak_threshold_rel
    edges = filter_by_min_distance(
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
    true_bits: np.ndarray,
    true_edges: np.ndarray,
) -> BitRecoveryResult:
    """Steps 10-15: segments -> bits -> metrics (undistorted M-point space)."""
    edges = edge_result.detected_edges
    bw    = edge_result.bit_width_px
    m     = len(edge_result.undistorted_signal)

    signal_start = 0.0
    signal_end   = float(m - 1)

    segments    = build_bit_segments(edges, bw, signal_start, signal_end)
    mean_period = _compute_mean_period(segments)

    rec_bits   = extract_bits_with_positions(segments)
    rec_values = np.array([b.value for b in rec_bits])

    best_err, accuracy = _sliding_window_accuracy(rec_values, true_bits)

    det_pos = np.array([e.position for e in edges])
    errs = []

    if det_pos.size > 0:
        det_sorted = np.sort(det_pos)
        used       = np.zeros(len(det_sorted), dtype=bool)
        half_bw    = bw * 0.5

        for te in true_edges:
            j = int(np.searchsorted(det_sorted, te))
            best_d, best_j = float("inf"), -1
            for jj in (j - 1, j):
                if 0 <= jj < len(det_sorted) and not used[jj]:
                    d = abs(det_sorted[jj] - te)
                    if d < best_d:
                        best_d, best_j = d, jj
            if best_j >= 0 and best_d < half_bw:
                errs.append(det_sorted[best_j] - te)
                used[best_j] = True

    errs_arr = np.array(errs)
    rms = float(np.sqrt(np.mean(errs_arr ** 2))) if errs_arr.size > 0 else 0.0

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
    true_bits: np.ndarray,
    true_edges: np.ndarray,
    config: Optional[BRConfig] = None,
    vignette: Optional[np.ndarray] = None,
) -> BitRecoveryResult:
    return recover_bits(
        detect_edges(adc_signal, config, vignette),
        true_bits,
        true_edges,
    )
