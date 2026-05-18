#!/usr/bin/env python3
"""
demo_opencv_br.py — OpenCV-визуализация алгоритма Blais-Rioux Edge Detector.

Структура окна (аналог demo_opencv.py из EncoderModel):
  ┌─────────────────────────────────────────────────────┐
  │  HEADER  — справочная текстовая информация          │
  ├─────────────────────────────────────────────────────┤
  │  PSEUDO-2D  — grayscale-полоса ПЗС-сигнала          │
  ├─────────────────────────────────────────────────────┤
  │  BIT PANEL  — нормализованный сигнал + разметка бит │
  └─────────────────────────────────────────────────────┘

CLI аналогичен demo_opencv.py: --source {sim|matrix}
  sim    — симуляция через ccd_simulator.py + blais_rioux.py
  matrix — реальная ПЗС-матрица через serial CSV (1+N значений на строку)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Попытка импортировать зависимости из пакета EncoderDecoder.
# Скрипт предполагается запускать из корня репозитория EncoderDecoder.
# ---------------------------------------------------------------------------
try:
    from src.ccd_simulator import SimulatorConfig, simulate_ccd
    from src.blais_rioux import BRConfig, detect_edges_and_recover_bits, EdgeDetectionResult
except ImportError:
    print(
        "[ERROR] Не найдены модули src.ccd_simulator / src.blais_rioux.\n"
        "Запускайте скрипт из корня репозитория EncoderDecoder:\n"
        "  python src/demo_opencv_br.py --source sim",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import serial as _serial_module
except ImportError:
    _serial_module = None  # type: ignore


# ============================================================
# Вспомогательные типы
# ============================================================

@dataclass
class FramePacket:
    """Один кадр данных с ПЗС-линейки."""
    row_index: int
    pixels: np.ndarray      # float32, длина = n_pixels
    source_label: str


@dataclass
class Layout:
    """Геометрия окна."""
    width: int
    header_h: int
    strip_h: int
    bit_h: int
    n_pixels: int           # длина сигнала


# ============================================================
# AutoRange: экспоненциально сглаженный диапазон
# ============================================================

class AutoRange:
    """EMA-скользящий диапазон для авто-масштабирования."""

    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self._low: Optional[float] = None
        self._high: Optional[float] = None

    def update(self, values: np.ndarray) -> Tuple[float, float]:
        lo = float(np.percentile(values, 1))
        hi = float(np.percentile(values, 99))
        if self._low is None:
            self._low, self._high = lo, hi
        else:
            a = self.alpha
            self._low  = (1 - a) * self._low  + a * lo
            self._high = (1 - a) * self._high + a * hi
        if self._high - self._low < 1e-6:
            self._high = self._low + 1.0
        return self._low, self._high


# ============================================================
# Утилиты рисования
# ============================================================

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN       # чуть уже для кода

COLOR_BG     = (245, 245, 245)
COLOR_TEXT   = (30,  30,  30)
COLOR_MUTED  = (120, 120, 120)
COLOR_ROI    = (180,  30,  30)           # BGR — тёмно-красный
COLOR_RISING = (20,  140, 220)           # BGR — синий (переход 0→1)
COLOR_FALL   = (200, 100,  20)           # BGR — оранжевый (переход 1→0)
COLOR_EDGE   = (100, 100, 100)
COLOR_OK     = (40,  160,  40)
COLOR_ERR    = (30,   30, 200)
COLOR_GRID   = (200, 200, 200)
COLOR_RECOV  = (220,  80,  20)           # BGR — оранжевый (восстан. бит)


def _put(img: np.ndarray, text: str, x: int, y: int,
         color=COLOR_TEXT, scale: float = 0.45, thickness: int = 1) -> None:
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _x_mapper(n_pixels: int, width: int, left_pad: int = 0):
    """Возвращает функцию перевода пиксель-ПЗС → x-координата на холсте."""
    plot_w = max(1, width - left_pad)

    def _map(px: float) -> int:
        t = float(np.clip(px / (n_pixels - 1), 0.0, 1.0))
        return int(round(left_pad + t * (plot_w - 1)))

    return _map


def normalize_pixels(pixels: np.ndarray, lo: float, hi: float,
                     invert: bool = False) -> np.ndarray:
    scaled = np.clip((pixels - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray = (scaled * 255).astype(np.uint8)
    return 255 - gray if invert else gray


# ============================================================
# Панели
# ============================================================

def draw_header(
    packet: FramePacket,
    det: Optional[EdgeDetectionResult],
    n_bits_apriori: int,
    lo: float, hi: float,
    width: int, height: int,
    filter_order: int,
    threshold_pct: float,
    has_ground_truth: bool = False,
    roi_left: int = 0,
    roi_right: int = 0,
) -> np.ndarray:
    """
    Справочная текстовая информация (аналог draw_header из demo_opencv).

    Параметр has_ground_truth=True только для --source sim, где истинная
    последовательность известна. При --source matrix метрики accuracy и
    rms_edge_error не вычисляются и заменяются на N/A.
    """
    hdr = np.full((height, width, 3), 245, dtype=np.uint8)

    period_err = 0.0
    if det is not None and det.bit_width_px > 0 and det.measured_bit_period > 0:
        period_err = (det.measured_bit_period - det.bit_width_px) / det.bit_width_px * 100.0

    lines: list[tuple[str, tuple, float, int]] = []

    # Строка 1: источник, строка, диапазон
    roi_clip_str = ""
    if roi_left > 0 or roi_right > 0:
        roi_clip_str = f"  roi_clip=[{roi_left}..n-{roi_right}]"
    lines.append((
        f"source={packet.source_label}  row={packet.row_index}"
        f"  range=[{lo:.1f}, {hi:.1f}]{roi_clip_str}",
        COLOR_MUTED, 0.44, 1
    ))

    if det is not None:
        # Строка 2: параметры алгоритма + ROI + период
        period_str = f"T={det.measured_bit_period:.3f} px ({period_err:+.2f}%)"
        lines.append((
            f"N={filter_order}  thr={threshold_pct:.0f}%  "
            f"roi=[{det.roi_start:.1f} .. {det.roi_end:.1f}]  {period_str}",
            COLOR_TEXT, 0.44, 1
        ))

        # Строка 3: детекция + метрики качества
        if has_ground_truth:
            # Режим симуляции — accuracy и rms доступны
            acc_color = (COLOR_OK if det.accuracy >= 99.0
                         else COLOR_ERR if det.accuracy < 90.0
                         else COLOR_MUTED)
            lines.append((
                f"edges={len(det.detected_edges)}  "
                f"bits_apriori={n_bits_apriori}  "
                f"bits_recov={len(det.recovered_bit_values)}  "
                f"rms={det.rms_edge_error:.3f} px  "
                f"acc={det.accuracy:.1f}%",
                acc_color, 0.44, 1
            ))
        else:
            # Режим матрицы — истинный сигнал неизвестен, accuracy/rms N/A
            n_exp  = n_bits_apriori
            n_got  = len(det.recovered_bit_values)
            # Оцениваем «совпадение» числа бит как косвенный индикатор
            count_color = COLOR_OK if n_got == n_exp else COLOR_ERR
            lines.append((
                f"edges={len(det.detected_edges)}  "
                f"bits_apriori={n_exp}  "
                f"bits_recov={n_got}  "
                f"rms=N/A  acc=N/A",
                count_color, 0.44, 1
            ))

        # Строка 4: восстановленная последовательность
        lines.append((
            f"Recov: {''.join(str(int(b)) for b in det.recovered_bit_values)}",
            COLOR_RECOV, 0.42, 1
        ))
    else:
        lines.append(("Processing...", COLOR_MUTED, 0.44, 1))

    y = 18
    for text, color, scale, thick in lines:
        _put(hdr, text, 8, y, color=color, scale=scale, thickness=thick)
        y += 18

    cv2.line(hdr, (0, height - 1), (width - 1, height - 1), COLOR_EDGE, 1)
    return hdr


def draw_strip_2d(pixels: np.ndarray, lo: float, hi: float,
                  width: int, height: int, invert: bool = False) -> np.ndarray:
    """
    Псевдо-2D grayscale-полоса — горизонтальная визуализация ПЗС-сигнала.
    Аналог верхней панели demo_opencv и ax_2d в visualizer.py.
    """
    gray = normalize_pixels(pixels, lo, hi, invert=invert)
    strip = np.repeat(gray[np.newaxis, :], height, axis=0)
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    if strip_bgr.shape[1] != width:
        strip_bgr = cv2.resize(strip_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
    return strip_bgr


def draw_bit_panel(
    pixels: np.ndarray,
    det: Optional[EdgeDetectionResult],
    width: int, height: int,
    n_pixels: int,
    lo: float, hi: float,
) -> np.ndarray:
    """
    Нижняя панель: нормализованный сигнал + разметка битов и фронтов.
    Аналог первого графика visualizer.py + draw_binary_panel из demo_opencv.

    Содержит:
    - Нормализованный сигнал (кривая)
    - Границы ROI (вертикальные линии)
    - Вертикальные линии фронтов (синие = 0→1, оранжевые = 1→0)
    - Горизонтальные отрезки восстановленных битов (bit_segments)
    - Метки «0» / «1» над каждым сегментом
    - Дробные границы ячеек (светло-серые, как в draw_binary_panel)
    """
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    xmap   = _x_mapper(n_pixels, width, left_pad=0)

    top_pad    = 4
    bot_pad    = 4
    plot_h     = height - top_pad - bot_pad
    sig_top    = top_pad + int(plot_h * 0.05)
    sig_bot    = top_pad + int(plot_h * 0.55)
    bit_y_hi   = top_pad + int(plot_h * 0.65)  # y для бит = 1
    bit_y_lo   = top_pad + int(plot_h * 0.90)  # y для бит = 0

    # --- Сигнал ---
    norm = np.clip((pixels - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    pts = []
    for i, v in enumerate(norm):
        y = int(round(sig_bot - v * (sig_bot - sig_top)))
        pts.append((xmap(i), y))
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)],
                  False, (80, 80, 80), 1, cv2.LINE_AA)

    if det is None:
        return canvas

    # --- ROI ---
    for roi_x in (det.roi_start, det.roi_end):
        xx = xmap(roi_x)
        cv2.line(canvas, (xx, top_pad), (xx, height - bot_pad),
                 COLOR_ROI, 1, cv2.LINE_AA)

    # --- Границы ячеек (дробные) — светло-серые ---
    centers = [seg.start_pos for seg in det.bit_segments]
    if det.bit_segments:
        centers.append(det.bit_segments[-1].end_pos)
    for cx in centers:
        xx = xmap(cx)
        cv2.line(canvas, (xx, bit_y_hi - 6), (xx, height - bot_pad),
                 COLOR_GRID, 1, cv2.LINE_AA)

    # --- Восстановленные биты (горизонтальные отрезки) ---
    for seg in det.bit_segments:
        x1 = xmap(seg.start_pos)
        x2 = xmap(seg.end_pos)
        y_seg = bit_y_hi if seg.bit_value else bit_y_lo
        cv2.line(canvas, (x1, y_seg), (x2, y_seg), COLOR_RECOV, 2, cv2.LINE_AA)
        # Метка «0» / «1»
        mid_x = (x1 + x2) // 2 - 4
        _put(canvas, str(seg.bit_value), mid_x, y_seg - 3,
             color=COLOR_RECOV, scale=0.38, thickness=1)

    # --- Фронты (вертикальные линии + кружки на кривой) ---
    for edge in det.detected_edges:
        xx = xmap(edge.position)
        color = COLOR_RISING if edge.d1_value > 0 else COLOR_FALL
        cv2.line(canvas, (xx, top_pad), (xx, height - bot_pad),
                 color, 1, cv2.LINE_AA)
        # Кружок на уровне сигнала
        idx = int(round(edge.position))
        if 0 <= idx < len(norm):
            cy = int(round(sig_bot - norm[idx] * (sig_bot - sig_top)))
            cv2.circle(canvas, (xx, cy), 3, color, -1, cv2.LINE_AA)

    # Рамка
    cv2.rectangle(canvas, (0, top_pad), (width - 1, height - bot_pad),
                  COLOR_EDGE, 1)
    # Метка панели
    _put(canvas, "Signal + bit recovery", 4, top_pad + 12, COLOR_MUTED, 0.38)
    return canvas



# ============================================================
# ROI-clipping wrapper
# ============================================================

def _detect_with_fixed_roi(
    pixels: np.ndarray,
    true_bits: np.ndarray,
    true_edges: np.ndarray,
    distort_coeff: float,
    br_config: "BRConfig",
    roi_left: int,
    roi_right: int,
) -> "EdgeDetectionResult":
    """
    Обрезает сигнал до фиксированного ROI [roi_left : n-roi_right],
    запускает детектор строго внутри этого окна, затем сдвигает
    все позиции фронтов/сегментов обратно в исходные координаты.
    """
    n = len(pixels)
    l = max(0, roi_left)
    r = max(0, roi_right)
    if l + r >= n:
        raise ValueError(
            f"roi_left={l} + roi_right={r} >= n_pixels={n}: ROI пустой"
        )

    # Срез сигнала
    clipped = pixels[l : n - r] if r > 0 else pixels[l:]
    clipped = clipped.astype(np.int32 if pixels.dtype.kind in "iu" else np.float64)

    # Истинные фронты внутри ROI (сдвигаем координаты)
    if len(true_edges):
        mask = (true_edges >= l) & (true_edges <= n - 1 - r)
        clipped_edges = true_edges[mask] - l
    else:
        clipped_edges = np.array([])

    det = detect_edges_and_recover_bits(
        clipped, true_bits, clipped_edges, distort_coeff, br_config
    )

    # Сдвиг координат обратно в пространство полного сигнала
    shift = float(l)

    # Фронты
    shifted_edges = []
    for e in det.detected_edges:
        from dataclasses import replace as _replace
        shifted_edges.append(_replace(e, position=e.position + shift))

    # Сегменты бит
    shifted_segs = []
    for seg in det.bit_segments:
        from dataclasses import replace as _replace
        shifted_segs.append(_replace(
            seg,
            start_pos=seg.start_pos + shift,
            end_pos=seg.end_pos + shift,
        ))

    # Восстановленные биты с позициями
    shifted_rbits = []
    for rb in det.recovered_bits:
        from dataclasses import replace as _replace
        shifted_rbits.append(_replace(rb, position=rb.position + shift))

    from dataclasses import replace as _replace
    br_config_clipped = _replace(
        br_config,
        roi_start=None,  # авто — детектор сам найдёт по фронтам внутри clipped
        roi_end=None,
    )

    det = detect_edges_and_recover_bits(
        clipped, true_bits, clipped_edges, distort_coeff, br_config_clipped
    )
    return det

# ============================================================
# Сборка кадра
# ============================================================

def compose_frame(
    packet: FramePacket,
    args,
    ranger: AutoRange,
    det: Optional[EdgeDetectionResult],
    filter_order: int,
    threshold_pct: float,
) -> np.ndarray:
    pixels = packet.pixels.astype(np.float32)
    lo, hi = (ranger.update(pixels)
               if args.min_val is None or args.max_val is None
               else (args.min_val, args.max_val))

    n_px   = len(pixels)
    width  = max(512, n_px * args.pixel_width)
    header_h = args.header_height
    strip_h  = max(30, args.strip_height)
    bit_h    = max(80, args.window_height - header_h - strip_h)

    has_gt = (packet.source_label == "simulation")
    hdr    = draw_header(packet, det, args.n_bits, lo, hi, width, header_h,
                         filter_order, threshold_pct, has_ground_truth=has_gt,
                         roi_left=args.roi_left, roi_right=args.roi_right)
    strip  = draw_strip_2d(pixels, lo, hi, width, strip_h, invert=args.invert)
    bp     = draw_bit_panel(pixels, det, width, bit_h, n_px, lo, hi)

    # Рисуем границы фиксированного ROI поверх strip и bit_panel
    xmap = _x_mapper(n_px, width)
    for clip_px, panel in ((args.roi_left, strip), (args.roi_left, bp),
                            (n_px - 1 - args.roi_right, strip), (n_px - 1 - args.roi_right, bp)):
        if clip_px <= 0 and panel is strip and args.roi_left == 0:
            continue
        if clip_px >= n_px - 1 and args.roi_right == 0:
            continue
        xx = xmap(clip_px)
        h  = panel.shape[0]
        cv2.line(panel, (xx, 0), (xx, h - 1), (0, 180, 80), 2, cv2.LINE_AA)

    return np.vstack([hdr, strip, bp])


# ============================================================
# Источники данных
# ============================================================

def matrix_packets(args):
    """Генератор пакетов из serial CSV (формат: row_index,px0,px1,...,pxN-1)."""
    if _serial_module is None:
        raise RuntimeError("pyserial не установлен: pip install pyserial")
    ser = _serial_module.Serial(args.port, args.baud, timeout=1)
    time.sleep(1.0)
    try:
        while True:
            line = ser.readline()
            packet = _parse_csv(line.decode("utf-8", errors="ignore"))
            if packet is not None:
                yield packet
    finally:
        ser.close()


def _parse_csv(text: str) -> Optional[FramePacket]:
    """Разбор строки CSV: row_index, px0, px1, ..."""
    try:
        parts = text.strip().split(",")
        if len(parts) < 2:
            return None
        row_idx = int(parts[0])
        pixels  = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        return FramePacket(row_index=row_idx, pixels=pixels, source_label="matrix")
    except Exception:
        return None


# ============================================================
# CLI
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "OpenCV-визуализация алгоритма Blais-Rioux Edge Detector.\n"
            "Аналог demo_opencv.py из EncoderModel, адаптированный для EncoderDecoder."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Источник ----
    src = p.add_argument_group("Источник данных")
    src.add_argument(
        "--source", choices=["sim", "matrix"], default="sim",
        help="Источник: sim=симуляция, matrix=serial ПЗС-матрица",
    )
    src.add_argument("--port",  default=None,   help="COM-порт (только --source=matrix)")
    src.add_argument("--baud",  type=int, default=115200, help="Скорость serial (default: 115200)")

    # ---- Параметры симуляции ----
    sim = p.add_argument_group("Параметры симуляции (--source sim)")
    sim.add_argument("--n-bits",    type=int,   default=24,    help="Количество бит (default: 24)")
    sim.add_argument("--bit-width", type=float, default=6,   help="Ширина бита, px (default: 6)")
    sim.add_argument("--blur",      type=float, default=0.5,   help="Размытие оптики σ (default: 0.5)")
    sim.add_argument("--noise",     type=float, default=60.0,  help="Шум АЦП ADU (default: 60)")
    sim.add_argument("--vignette",  type=float, default=0.25,  help="Виньетирование 0..1 (default: 0.25)")
    sim.add_argument("--distort",   type=float, default=0.0,   help="Дисторсия (default: 0.0)")
    sim.add_argument("--n-pixels",  type=int,   default=128,   help="Пикселей ПЗС (default: 128)")
    sim.add_argument("--seed",      type=int,   default=7,     help="Random seed (default: 7)")
    sim.add_argument(
        "--animate", action="store_true",
        help="Обновлять seed каждый кадр (анимированная демонстрация)",
    )
    sim.add_argument("--fps", type=float, default=2.0, help="Частота смены кадров для --animate (default: 2)")

    # ---- Параметры алгоритма Blais-Rioux ----
    br = p.add_argument_group("Алгоритм Blais-Rioux")
    br.add_argument(
        "--filter-order", type=int, choices=[2, 4], default=2,
        help="Порядок КИХ-фильтра N=2 или N=4 (default: 2)",
    )
    br.add_argument(
        "--threshold", type=float, default=15.0,
        help="Порог |D1| в %% от максимума (default: 15)",
    )
    br.add_argument(
        "--min-dist", type=float, default=30.0,
        help="Мин. расстояние между фронтами в %% от T (default: 30)",
    )
    br.add_argument(
        "--smoothing", type=float, default=0.2,
        help="Сглаживание σ перед D1 (default: 0.2)",
    )
    br.add_argument(
        "--minmax-window", type=int, default=50,
        help="Окно min-max нормировки, px (default: 50)",
    )


    # ---- Параметры отображения ----
    disp = p.add_argument_group("Параметры отображения")
    disp.add_argument("--roi-left",      type=int,   default=0,   help="Обрезка сигнала слева, пикселей (default: 0)")
    disp.add_argument("--roi-right",     type=int,   default=0,   help="Обрезка сигнала справа, пикселей (default: 0)")
    disp.add_argument("--pixel-width",   type=int,   default=5,   help="Ширина одного пикселя ПЗС на экране (default: 5)")
    disp.add_argument("--header-height", type=int,   default=88,  help="Высота заголовка, px (default: 88)")
    disp.add_argument("--strip-height",  type=int,   default=40,  help="Высота 2D-полосы, px (default: 40)")
    disp.add_argument("--window-height", type=int,   default=500, help="Общая высота окна (default: 500)")
    disp.add_argument("--invert",        action="store_true",     help="Инвертировать grayscale")
    disp.add_argument("--min",           dest="min_val", type=float, default=None, help="Фиксированный минимум")
    disp.add_argument("--max",           dest="max_val", type=float, default=None, help="Фиксированный максимум")
    disp.add_argument("--save",          default=None, help="Сохранить кадр в PNG и выйти")
    disp.add_argument("--no-window",     action="store_true",     help="Не показывать окно (только --save)")

    return p


# ============================================================
# Главный цикл
# ============================================================

def run(args) -> None:
    br_config = BRConfig(
        filter_order=args.filter_order,
        peak_threshold_rel=args.threshold / 100.0,
        min_edge_distance_factor=args.min_dist / 100.0,
        bit_width_px=args.bit_width,
        smoothing_sigma=args.smoothing,
        minmax_window_px=args.minmax_window,
    )

    ranger = AutoRange()
    win    = "Blais-Rioux Edge Detector"

    # ---------- source=matrix: поточный режим ----------
    if args.source == "matrix":
        if args.port is None:
            print("[ERROR] --port не задан для --source=matrix", file=sys.stderr)
            sys.exit(1)
        if not args.no_window:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        for packet in matrix_packets(args):
            pixels = packet.pixels
            # ground truth неизвестен при работе с матрицей → фиктивные данные
            true_bits  = np.zeros(args.n_bits, dtype=np.int32)
            true_edges = np.array([])
            try:
                det = _detect_with_fixed_roi(
                    pixels,
                    true_bits, true_edges,
                    0.0,
                    br_config,
                    roi_left=args.roi_left,
                    roi_right=args.roi_right,
                )
            except Exception as e:
                print(f"[WARN] Ошибка обработки: {e}")
                det = None

            frame = compose_frame(packet, args, ranger, det,
                                  args.filter_order, args.threshold)
            if args.save:
                cv2.imwrite(args.save, frame)
                print(f"Сохранено: {args.save}")
                break
            if not args.no_window:
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        if not args.no_window:
            cv2.destroyAllWindows()
        return

    # ---------- source=sim ----------
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
            distort_coeff=args.distort,
            n_pixels=args.n_pixels,
            seed=seed,
        )
        sim_result = simulate_ccd(sim_cfg)
        packet = FramePacket(
            row_index=seed if args.animate else 0,
            pixels=sim_result.adc_signal.astype(np.float32),
            source_label="simulation",
        )

        try:
            det = _detect_with_fixed_roi(
                sim_result.adc_signal.astype(np.float64),
                sim_result.bits,
                sim_result.true_edges,
                sim_cfg.distort_coeff,
                br_config,
                roi_left=args.roi_left,
                roi_right=args.roi_right,
            )
        except Exception as e:
            print(f"[WARN] Ошибка детектора: {e}")
            det = None

        frame = compose_frame(packet, args, ranger, det,
                              args.filter_order, args.threshold)

        if args.save:
            cv2.imwrite(args.save, frame)
            print(f"Сохранено: {args.save}")
            break

        if args.no_window:
            break

        cv2.imshow(win, frame)

        wait = frame_interval if args.animate else 0
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            seed = (seed + 1) if args.animate else seed
            ranger = AutoRange()

        if args.animate:
            seed += 1
        else:
            # Статичный кадр — ждём нажатия
            pass

    if not args.no_window:
        cv2.destroyAllWindows()


def main() -> None:
    parser = build_argparser()
    args   = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
