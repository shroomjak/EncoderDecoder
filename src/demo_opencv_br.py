#!/usr/bin/env python3
"""
demo_opencv_br.py — OpenCV visualisation of the Blais-Rioux Edge Detector
with 1-D distortion correction.

Window layout
-------------
┌──────────────────────────────────────────────┐
│ HEADER — stats, k, angle, recovered bits    │
├──────────────────────────────────────────────┤
│ ORIGINAL strip (raw ADC, centred)           │
├──────────────────────────────────────────────┤
│ CORRECTED strip (after undistortion)        │
├──────────────────────────────────────────────┤
│ BIT PANEL — undistorted signal + bit marks  │
└──────────────────────────────────────────────┘
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import serial as _serial_module

from src.ccd_simulator import SimulatorConfig, simulate_ccd
from src.blais_rioux import (
    BRConfig,
    BitRecoveryResult,
    detect_edges,
    recover_bits,
)
from src.disk_angle_estimator import (
    AngleEstimationResult,
    build_code_angle_map,
    estimate_disk_angle_from_result, CODEWORD_LENGTH_BITS,
    TOTAL_CODE_BITS_ON_DISK, ANGLE_PERIOD_DEG, FULL_DISK_CODE_SEQUENCE,
)
from src.distortion.math1d import restore_signal_1d


SENSOR_CENTER_PIXEL = None
ANGLE_PER_SENSOR_PIXEL_DEG = None
ENABLE_ANGLE_ESTIMATION = True
DISTORT_K: float = 0.05


@dataclass
class FramePacket:
    row_index: int
    pixels: np.ndarray
    source_label: str


class AutoRange:
    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self._lo: Optional[float] = None
        self._hi: Optional[float] = None

    def update(self, values: np.ndarray) -> Tuple[float, float]:
        lo = float(np.percentile(values, 1))
        hi = float(np.percentile(values, 99))
        if self._lo is None:
            self._lo, self._hi = lo, hi
        else:
            a = self.alpha
            self._lo = (1 - a) * self._lo + a * lo
            self._hi = (1 - a) * self._hi + a * hi
        if self._hi - self._lo < 1e-6:
            self._hi = self._lo + 1.0
        return self._lo, self._hi


FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_TEXT = (30, 30, 30)
COLOR_MUTED = (120, 120, 120)
COLOR_ROI = (180, 30, 30)
COLOR_RISE = (20, 140, 220)
COLOR_FALL = (200, 100, 20)
COLOR_EDGE = (100, 100, 100)
COLOR_OK = (40, 160, 40)
COLOR_ERR = (30, 30, 200)
COLOR_GRID = (200, 200, 200)
COLOR_REC = (220, 80, 20)
COLOR_UNDIST_LABEL = (20, 120, 20)


def _put(img, text, x, y, color=COLOR_TEXT, scale=0.45, thick=1):
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _x_mapper(n_pixels: int, width: int, left_pad: int = 0):
    plot_w = max(1, width - left_pad)
    denom = max(1, n_pixels - 1)

    def _map(px: float) -> int:
        t = float(np.clip(px / denom, 0.0, 1.0))
        return int(round(left_pad + t * (plot_w - 1)))

    return _map


def normalize_pixels(pixels: np.ndarray, lo: float, hi: float, invert: bool = False) -> np.ndarray:
    scaled = np.clip((pixels - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray = (scaled * 255).astype(np.uint8)
    return 255 - gray if invert else gray


def draw_strip_2d(
    pixels: np.ndarray,
    lo: float,
    hi: float,
    canvas_width: int,
    height: int,
    invert: bool = False,
    natural_width: int | None = None,
    label: str = "",
    label_color=COLOR_MUTED,
) -> np.ndarray:
    gray = normalize_pixels(np.asarray(pixels, dtype=np.float32), lo, hi, invert=invert)
    render_w = canvas_width if natural_width is None else int(natural_width)
    render_w = int(np.clip(render_w, 1, canvas_width))

    strip = np.repeat(gray[np.newaxis, :], height, axis=0)
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    if strip_bgr.shape[1] != render_w:
        strip_bgr = cv2.resize(strip_bgr, (render_w, height), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((height, canvas_width, 3), dtype=np.uint8)
    x0 = max(0, (canvas_width - render_w) // 2)
    canvas[:, x0:x0 + render_w] = strip_bgr[:, :render_w]

    if label:
        lx = x0 + 4 if natural_width is not None else 4
        _put(canvas, label, lx, 14, color=label_color, scale=0.4)
    return canvas


def draw_header(
    packet, det, angle_est, n_bits_apriori,
    lo, hi, width, height,
    filter_order, threshold_pct, distort_k,
    has_ground_truth: bool = False,
) -> np.ndarray:
    hdr = np.full((height, width, 3), 245, dtype=np.uint8)
    lines = []
    lines.append((
        f"source={packet.source_label} row={packet.row_index} "
        f"range=[{lo:.1f},{hi:.1f}] k={distort_k:.5f}",
        COLOR_MUTED, 0.44, 1,
    ))

    if det is None:
        lines.append(("Processing...", COLOR_MUTED, 0.44, 1))
    else:
        period_err = 0.0
        if det.bit_width_px > 0 and det.measured_bit_period > 0:
            period_err = (det.measured_bit_period - det.bit_width_px) / det.bit_width_px * 100.0
        lines.append((
            f"N={filter_order} thr={threshold_pct:.0f}% T={det.measured_bit_period:.3f}px ({period_err:+.2f}%)",
            COLOR_TEXT, 0.44, 1,
        ))
        n_exp, n_got = n_bits_apriori, len(det.recovered_bit_values)
        if has_ground_truth:
            col = COLOR_OK if det.accuracy >= 99.0 else (COLOR_ERR if det.accuracy < 90.0 else COLOR_MUTED)
            lines.append((
                f"edges={len(det.detected_edges)} bits_apriori={n_exp} bits_recov={n_got} rms={det.rms_edge_error:.3f}px acc={det.accuracy:.1f}%",
                col, 0.44, 1,
            ))
        else:
            col = COLOR_OK if n_got == n_exp else COLOR_ERR
            lines.append((
                f"edges={len(det.detected_edges)} bits_apriori={n_exp} bits_recov={n_got} rms=N/A acc=N/A",
                col, 0.44, 1,
            ))
        if angle_est is None:
            lines.append(("Angle: N/A", COLOR_MUTED, 0.44, 1))
        elif angle_est.mean_angle_deg is None:
            lines.append((
                f"Angle: N/A full_bits={angle_est.visible_bits} sync={angle_est.matched_windows}/{angle_est.total_windows}",
                COLOR_MUTED, 0.44, 1,
            ))
        else:
            std_s = f"{angle_est.std_angle_deg:.5f} deg" if angle_est.std_angle_deg is not None else "N/A"
            lines.append((
                f"Angle: {angle_est.mean_angle_deg:.4f} deg, std={std_s}, full_bits={angle_est.visible_bits} sync={angle_est.matched_windows}/{angle_est.total_windows}",
                COLOR_TEXT, 0.44, 1,
            ))
        lines.append((
            f"Recov: {''.join(str(int(b)) for b in det.recovered_bit_values)}",
            COLOR_REC, 0.42, 1,
        ))

    y = 18
    for text, color, scale, thick in lines:
        _put(hdr, text, 8, y, color=color, scale=scale, thick=thick)
        y += 18
    cv2.line(hdr, (0, height - 1), (width - 1, height - 1), COLOR_EDGE, 1)
    return hdr


def draw_bit_panel(
    pixels: np.ndarray,
    det: Optional[BitRecoveryResult],
    width: int,
    height: int,
    n_pixels: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    xmap = _x_mapper(n_pixels, width)
    top_pad, bot_pad = 4, 4
    plot_h = height - top_pad - bot_pad
    sig_top = top_pad + int(plot_h * 0.05)
    sig_bot = top_pad + int(plot_h * 0.55)
    bit_y_hi = top_pad + int(plot_h * 0.65)
    bit_y_lo = top_pad + int(plot_h * 0.90)

    norm = np.clip((pixels - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    pts = [(xmap(i), int(round(sig_bot - v * (sig_bot - sig_top)))) for i, v in enumerate(norm)]
    if pts:
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, (80, 80, 80), 1, cv2.LINE_AA)

    if det is not None:
        vignette_norm = det.edge_result.vignette_norm
        pts2 = [(xmap(i), int(round(sig_bot - v * (sig_bot - sig_top)))) for i, v in enumerate(vignette_norm)]
        if pts2:
            cv2.polylines(canvas, [np.array(pts2, np.int32)], False, (150, 20, 20), 1, cv2.LINE_AA)

    if det is None:
        cv2.rectangle(canvas, (0, top_pad), (width - 1, height - bot_pad), COLOR_EDGE, 1)
        _put(canvas, "Signal (undistorted) + full-bit recovery", 4, top_pad + 12, COLOR_MUTED, 0.38)
        return canvas

    full_cells, boundaries = [], []
    for rb in det.recovered_bits:
        if rb.segment_idx < 0 or rb.segment_idx >= len(det.bit_segments):
            continue
        seg = det.bit_segments[rb.segment_idx]
        if seg.measured_period <= 1e-12:
            continue
        bs = float(rb.position)
        be = bs + seg.measured_period
        eps = max(1e-9, 1e-6 * seg.measured_period)
        if bs >= seg.start_pos - eps and be <= seg.end_pos + eps:
            full_cells.append((bs, be, int(rb.value)))
            boundaries += [bs, be]
    if boundaries:
        uniq = []
        for b in sorted(boundaries):
            if not uniq or abs(b - uniq[-1]) > 1e-6:
                uniq.append(b)
        for bx in uniq:
            cv2.line(canvas, (xmap(bx), bit_y_hi - 6), (xmap(bx), height - bot_pad), COLOR_GRID, 1, cv2.LINE_AA)

    for bs, be, bv in full_cells:
        x1, x2 = xmap(bs), xmap(be)
        yy = bit_y_hi if bv else bit_y_lo
        cv2.line(canvas, (x1, yy), (x2, yy), COLOR_REC, 2, cv2.LINE_AA)
        if abs(x2 - x1) >= 8:
            _put(canvas, str(bv), (x1 + x2) // 2 - 4, yy - 3, color=COLOR_REC, scale=0.38)

    for edge in det.detected_edges:
        xx = xmap(edge.position)
        col = COLOR_RISE if edge.d1_value > 0 else COLOR_FALL
        cv2.line(canvas, (xx, top_pad), (xx, height - bot_pad), col, 1, cv2.LINE_AA)
        idx = int(round(edge.position))
        if 0 <= idx < len(norm):
            cy = int(round(sig_bot - norm[idx] * (sig_bot - sig_top)))
            cv2.circle(canvas, (xx, cy), 3, col, -1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, top_pad), (width - 1, height - bot_pad), COLOR_EDGE, 1)
    _put(canvas, "Signal (undistorted) + full-bit recovery", 4, top_pad + 12, COLOR_MUTED, 0.38)
    return canvas


def _detect_full_width(
    pixels: np.ndarray,
    true_bits: np.ndarray,
    true_edges: np.ndarray,
    br_config: BRConfig,
) -> BitRecoveryResult:
    """Detect edges and recover bits over the full width of the undistorted signal."""
    er = detect_edges(np.asarray(pixels, dtype=np.float64), br_config)
    return recover_bits(er, true_bits, true_edges)


def _estimate_angle_if_possible(
    det: Optional[BitRecoveryResult],
) -> Optional[AngleEstimationResult]:
    if not ENABLE_ANGLE_ESTIMATION or det is None:
        return None
    if CODEWORD_LENGTH_BITS <= 0 or TOTAL_CODE_BITS_ON_DISK <= 0:
        return None

    # sensor_width_px и sensor_center_px — оба в координатах undistorted сигнала
    n_undist = len(det.edge_result.undistorted_signal)
    sensor_width_px = float(n_undist)
    sensor_center_px = (
        float(SENSOR_CENTER_PIXEL)
        if SENSOR_CENTER_PIXEL is not None
        else 0.5 * (n_undist - 1)
    )

    return estimate_disk_angle_from_result(
        detection_result=det,
        code_sequence=FULL_DISK_CODE_SEQUENCE,
        codeword_length=CODEWORD_LENGTH_BITS,
        total_code_bits=TOTAL_CODE_BITS_ON_DISK,
        sensor_width_px=sensor_width_px,
        sensor_center_px=sensor_center_px,
        angle_period_deg=ANGLE_PERIOD_DEG,
    )


def compose_frame(
    packet: FramePacket,
    args,
    ranger: AutoRange,
    det: Optional[BitRecoveryResult],
    angle_est: Optional[AngleEstimationResult],
    filter_order: int,
    threshold_pct: float,
    distort_k: float,
) -> np.ndarray:
    raw_pixels = np.asarray(packet.pixels, dtype=np.float32)

    if args.min_val is not None and args.max_val is not None:
        lo, hi = args.min_val, args.max_val
    else:
        lo, hi = ranger.update(raw_pixels)

    if det is not None:
        vis_undist = np.asarray(det.undistorted_signal, dtype=np.float32)
    elif abs(distort_k) > 1e-15:
        _, vis_undist, _ = restore_signal_1d(
            raw_pixels.astype(np.float64),
            distort_k,
            output_step_px=1.0,
            clip_to_adc=True,
            as_uint16=False,
        )
        vis_undist = np.asarray(vis_undist, dtype=np.float32)
    else:
        vis_undist = raw_pixels.copy()

    n_raw = len(raw_pixels)
    n_undist = len(vis_undist)
    hdr_h = args.header_height
    str_h = max(30, args.strip_height)
    bit_h = max(80, args.window_height - hdr_h - 2 * str_h - 4)
    has_gt = packet.source_label == "simulation"
    # Ширина холста определяется undistorted сигналом
    width = max(512, n_undist * args.pixel_width)
    # ORIGINAL отображается в своём натуральном масштабе пикселя
    raw_natural_w = n_raw * args.pixel_width  # ← было: round(width * n_raw / n_undist)

    hdr = draw_header(
        packet, det, angle_est, args.n_bits, lo, hi,
        width, hdr_h, filter_order, threshold_pct, distort_k,
        has_ground_truth=has_gt
    )

    # При вызовах draw_strip_2d оба получают свой natural_width:
    s_raw = draw_strip_2d(
        raw_pixels, lo, hi, width, str_h,
        invert=args.invert,
        natural_width=raw_natural_w,  # пиксели ORIGINAL в истинном масштабе
        label="ORIGINAL", label_color=COLOR_MUTED,
    )
    s_undist = draw_strip_2d(
        vis_undist, lo, hi, width, str_h,
        invert=args.invert,
        natural_width=width,  # undistorted занимает весь холст
        label=f"CORRECTED (k={distort_k:.4f})", label_color=COLOR_UNDIST_LABEL,
    )
    bp = draw_bit_panel(
        vis_undist,
        det,
        width,
        bit_h,
        n_pixels=n_undist,
        lo=lo,
        hi=hi,
    )

    div = np.full((2, width, 3), 180, dtype=np.uint8)
    return np.vstack([hdr, s_raw, div, s_undist, div, bp])


def matrix_packets(args):
    ser = _serial_module.Serial(args.port, args.baud, timeout=1)
    time.sleep(1.0)
    try:
        while True:
            line = ser.readline()
            pkt = _parse_csv(line.decode("utf-8", errors="ignore"))
            if pkt is not None:
                yield pkt
    finally:
        ser.close()


def _parse_csv(text: str) -> Optional[FramePacket]:
    try:
        parts = text.strip().split(",")
        if len(parts) < 2:
            return None
        return FramePacket(
            row_index=int(parts[0]),
            pixels=np.array(parts[1:], dtype=np.float32),
            source_label="matrix",
        )
    except Exception:
        return None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenCV Blais-Rioux Edge Detector with distortion correction.")
    src = p.add_argument_group("Data source")
    src.add_argument("--source", choices=["sim", "matrix"], default="sim")
    src.add_argument("--port", default=None)
    src.add_argument("--baud", type=int, default=115200)

    sim = p.add_argument_group("Simulation")
    sim.add_argument("--n-bits", type=int, default=24)
    sim.add_argument("--bit-width", type=float, default=6)
    sim.add_argument("--blur", type=float, default=0.5)
    sim.add_argument("--noise", type=float, default=60.0)
    sim.add_argument("--vignette", type=float, default=0.25)
    sim.add_argument("--n-pixels", type=int, default=128)
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--animate", action="store_true")
    sim.add_argument("--fps", type=float, default=2.0)

    br = p.add_argument_group("Blais-Rioux algorithm")
    br.add_argument("--filter-order", type=int, choices=[2, 4], default=2)
    br.add_argument("--threshold", type=float, default=15.0)
    br.add_argument("--min-dist", type=float, default=30.0)
    br.add_argument("--smoothing", type=float, default=0.2)

    cor = p.add_argument_group("Distortion correction")
    cor.add_argument("--distort-k", type=float, default=None)

    disp = p.add_argument_group("Display")
    disp.add_argument("--pixel-width", type=int, default=5)
    disp.add_argument("--header-height", type=int, default=120)
    disp.add_argument("--strip-height", type=int, default=40)
    disp.add_argument("--window-height", type=int, default=600)
    disp.add_argument("--invert", action="store_true")
    disp.add_argument("--min", dest="min_val", type=float, default=None)
    disp.add_argument("--max", dest="max_val", type=float, default=None)
    disp.add_argument("--save", default=None)
    disp.add_argument("--no-window", action="store_true")
    return p


def run(args) -> None:
    distort_k = args.distort_k if args.distort_k is not None else DISTORT_K
    br_config = BRConfig(
        filter_order=args.filter_order,
        peak_threshold_rel=args.threshold / 100.0,
        min_edge_distance_factor=args.min_dist / 100.0,
        bit_width_px=args.bit_width,
        smoothing_sigma=args.smoothing,
        distort_coeff=distort_k,
    )

    ranger = AutoRange()
    win = "Blais-Rioux Edge Detector"

    if args.source == "matrix":
        if args.port is None:
            print("[ERROR] --port required for --source=matrix", file=sys.stderr)
            sys.exit(1)
        if not args.no_window:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        for packet in matrix_packets(args):
            det = angle_est = None
            try:
                det = _detect_full_width(
                    packet.pixels,
                    np.zeros(args.n_bits, dtype=np.int32),
                    np.array([]),
                    br_config,
                )
                angle_est = _estimate_angle_if_possible(det)
            except Exception as e:
                print(f"[WARN] {e}")
            frame = compose_frame(packet, args, ranger, det, angle_est, args.filter_order, args.threshold, distort_k)
            if args.save:
                cv2.imwrite(args.save, frame)
                break
            if not args.no_window:
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        if not args.no_window:
            cv2.destroyAllWindows()
        return

    seed = args.seed
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frame_interval = max(1, int(1000 / max(args.fps, 0.1))) if args.animate else 0

    while True:
        sim_cfg = SimulatorConfig(
            n_bits=args.n_bits,
            bit_width_px=args.bit_width,
            sigma_blur_px=args.blur,
            noise_sigma_adu=args.noise,
            vignette_strength=args.vignette,
            distort_coeff=distort_k,
            n_pixels=args.n_pixels,
            seed=seed,
        )
        sim_result = simulate_ccd(sim_cfg)
        packet = FramePacket(
            row_index=seed if args.animate else 0,
            pixels=sim_result.adc_signal.astype(np.float32),
            source_label="simulation",
        )
        det = angle_est = None
        try:
            det = _detect_full_width(
                sim_result.adc_signal.astype(np.float64),
                sim_result.bits,
                sim_result.true_edges,
                br_config,
            )
            angle_est = _estimate_angle_if_possible(det)
        except Exception as e:
            print(f"[WARN] Detector error: {e}")

        frame = compose_frame(packet, args, ranger, det, angle_est, args.filter_order, args.threshold, distort_k)
        if args.save:
            cv2.imwrite(args.save, frame)
            break
        if args.no_window:
            break
        cv2.imshow(win, frame)
        key = cv2.waitKey(frame_interval if args.animate else 0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            ranger = AutoRange()
        if args.animate:
            seed += 1

    if not args.no_window:
        cv2.destroyAllWindows()


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
