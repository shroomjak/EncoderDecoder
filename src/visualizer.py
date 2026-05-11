"""
visualizer.py — Модуль визуализации через Matplotlib

Создает фигуру с пятью секциями:
1. Header: диагностическая информация
2. Псевдо-2D изображение сигнала (оттенки серого)
3. Нормализованный, сглаженный и восстановленный сигнал
4. Первая производная D1 с обозначением фронтов
5. Вторая производная D2 с обозначением нулей (фронтов)

Панели 3, 4, 5 содержат разметку ROI и истинных положений бит.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from typing import Tuple, Optional

from src.ccd_simulator import SimulatorConfig, SimulationResult, simulate_ccd
from src.blais_rioux import (
    BRConfig, EdgeDetectionResult, DetectedEdge,
    detect_edges_and_recover_bits,
)

# Цветовая гамма (matplotlib defaults)
C0 = "#1f77b4"   # синий
C1 = "#ff7f0e"   # оранжевый
C2 = "#2ca02c"   # зелёный
C4 = "#9467bd"   # фиолетовый
C5 = "#8c564b"   # коричневый
C7 = "#7f7f7f"   # серый

COLOR_SIGNAL      = C0
COLOR_SMOOTHED    = C7
COLOR_LOCAL_NORM  = C0
COLOR_RECOVERED   = C1
COLOR_D1          = C0
COLOR_D2          = C4
COLOR_TRUE_BIT    = "#c0c0c0"    # светло-серая пунктирная сетка истинных бит
COLOR_ROI         = C0
COLOR_THRESHOLD   = C5
COLOR_RISING      = C1           # фронт 0→1
COLOR_FALLING     = C4           # фронт 1→0
COLOR_CORRECT     = C2
COLOR_ERROR_COL   = "#d62728"


# Вспомогательные функции
def bits_to_string(bits: np.ndarray) -> str:
    return "".join(str(int(b)) for b in bits)


def _add_roi_and_true_bits(
    ax: plt.Axes,
    det: "EdgeDetectionResult",
    true_bit_centers: np.ndarray,
) -> None:
    """Добавляет вертикальные линии ROI и истинных центров бит на ось."""
    # Истинные центры бит (пунктир, очень светлые)
    for cx in true_bit_centers:
        ax.axvline(cx, color=COLOR_TRUE_BIT, lw=0.7, ls=(0, (3, 6)), zorder=1)
    # ROI
    ax.axvline(det.roi_start, color=COLOR_ROI, lw=1.0, ls="--", alpha=0.8, zorder=2)
    ax.axvline(det.roi_end,   color=COLOR_ROI, lw=1.0, ls="--", alpha=0.8, zorder=2)


def _add_edge_markers(
    ax: plt.Axes,
    det: "EdgeDetectionResult",
    draw_dot: bool = False,
    dot_y: Optional[float] = None,
) -> None:
    """Рисует вертикальные линии фронтов на оси."""
    for edge in det.detected_edges:
        color = COLOR_RISING if edge.d1_value > 0 else COLOR_FALLING
        ax.axvline(edge.position, color=color, lw=1.0, alpha=0.85, zorder=3)
        if draw_dot and dot_y is not None:
            ax.plot(edge.position, dot_y, "o", color=color, ms=4, zorder=4)


# Основная функция визуализации
def visualize_result(
    sim_result: SimulationResult,
    detection_result: EdgeDetectionResult,
    window_name: str = "Цифровая модель",
) -> plt.Figure:
    """
    Создает фигуру с визуализацией результатов.

    Parameters
    ----------
    sim_result : SimulationResult
        Результат симуляции ПЗС.
    detection_result : EdgeDetectionResult
        Результат детектирования фронтов.
    window_name : str
        Заголовок окна / фигуры.

    Returns
    -------
    plt.Figure
        Фигура matplotlib.
    """
    config = sim_result.config
    det    = detection_result
    n_px   = config.n_pixels
    xs     = np.arange(n_px)

    # --- Компоновка ---
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.canvas.manager.set_window_title(window_name) if hasattr(fig.canvas, "manager") else None

    gs = gridspec.GridSpec(
        5, 1,
        figure=fig,
        height_ratios=[0.55, 0.30, 1.0, 1.0, 1.0],
        hspace=0.08,
        left=0.06, right=0.97, top=0.97, bottom=0.06,
    )

    # ===================================================================
    # 1. HEADER — диагностическая информация
    # ===================================================================
    ax_hdr = fig.add_subplot(gs[0])
    ax_hdr.axis("off")

    period_err = (
        (det.measured_bit_period - det.bit_width_px) / det.bit_width_px * 100
        if det.measured_bit_period > 0 else 0.0
    )
    acc_color = (
        COLOR_CORRECT if det.accuracy >= 99.0
        else (C5 if det.accuracy >= 90.0 else COLOR_ERROR_COL)
    )

    lines = [
        (
            f"Бит: {config.n_bits}  |  Ширина: {config.bit_width_px:.1f} px  |  "
            f"Сглаж.: {config.sigma_blur_px:.1f} px  |  Шум: {config.noise_sigma_adu:.0f} ADU  |  "
            f"Виньет.: {config.vignette_strength:.2f}  |  Seed: {config.seed}",
            "dimgray", 14,
        ),
        (
            f"Порядок N={det.config.filter_order}  |  "
            f"Гран. нуля: {det.config.peak_threshold_rel*100:.0f}%  |  "
            f"Мин. дист.: {det.config.min_edge_distance_factor*100:.0f}%T  |  "
            f"Сглаж.: {det.config.smoothing_sigma:.1f} px  |  "
            f"Окно норм.: {det.config.minmax_window_px} px",
            "dimgray", 14,
        ),
        (
            f"Точн.: {det.accuracy:.1f}%  |  Перех.: {len(det.detected_edges)}  |  "
            f"RMS: {det.rms_edge_error:.3f} px  |  "
            f"Ср. T: {det.measured_bit_period:.2f} px ({period_err:+.2f}%)  |  "
            f"ROI: [{det.roi_start:.1f} – {det.roi_end:.1f}]",
            acc_color, 14,
        ),
        (f"{'Истинное:'.ljust(20)} {bits_to_string(sim_result.bits)}", COLOR_CORRECT, 12),
        (f"{'Восстановленное:'.ljust(20)} {bits_to_string(det.recovered_bit_values)}", COLOR_RECOVERED, 12),
    ]
    y_pos = 0.98
    step  = 0.18
    for text, color, fsize in lines:
        ax_hdr.text(
            0.0, y_pos, text,
            transform=ax_hdr.transAxes,
            fontsize=fsize, color=color,
            va="top", ha="left",
            fontfamily="monospace",
        )
        y_pos -= step

    # ===================================================================
    # 2. Псевдо-2D изображение сигнала
    # ===================================================================
    ax_2d = fig.add_subplot(gs[1])

    sig_raw = sim_result.adc_signal.astype(float)
    sig_norm = (sig_raw - sig_raw.min()) / (sig_raw.max() - sig_raw.min() + 1e-12)
    # Растягиваем вертикально (pseudo-2D strip)
    strip = np.tile(sig_norm[np.newaxis, :], (20, 1))
    ax_2d.imshow(strip, cmap="gray", aspect="auto",
                 extent=[-0.5, n_px - 0.5, 0, 1],
                 origin="upper", vmin=0, vmax=1)
    ax_2d.set_yticks([])
    ax_2d.set_xticks([])
    for spine in ax_2d.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)
    ax_2d.set_ylabel("ПЗС", fontsize=10, rotation=0, labelpad=26, va="center")

    # ===================================================================
    # 3. Нормализованный + восстановленный сигнал
    # ===================================================================
    ax_sig = fig.add_subplot(gs[2])

    _add_roi_and_true_bits(ax_sig, det, sim_result.true_bit_centers)

    ax_sig.plot(xs, det.local_norm_signal, color=COLOR_LOCAL_NORM, lw=1.4, label="LocalNorm", zorder=6)

    # Восстановленные уровни бит
    for seg in det.bit_segments:
        y_lvl = 0.1 + seg.bit_value * 0.8
        ax_sig.hlines(y_lvl, seg.start_pos, seg.end_pos,
                      colors=COLOR_RECOVERED, lw=2.5, alpha=0.9, zorder=7)

    _add_edge_markers(ax_sig, det)

    ax_sig.set_ylim(-0.05, 1.15)
    ax_sig.set_xlim(-0.5, n_px - 0.5)
    ax_sig.set_xticks([])
    ax_sig.set_yticks([0, 0.5, 1.0])
    ax_sig.yaxis.set_tick_params(labelsize=7)
    ax_sig.set_ylabel("Сигнал", fontsize=10)
    ax_sig.grid(axis="y", ls=":", lw=0.5, alpha=0.4)

    # Легенда
    legend_items = [
        Line2D([0], [0], color=COLOR_LOCAL_NORM, lw=1.4, label="Нормал."),
        Line2D([0], [0], color=COLOR_RECOVERED,  lw=2.5, label="Восстан."),
        Line2D([0], [0], color=COLOR_ROI,        lw=1.0, ls="--", label="ROI"),
        Line2D([0], [0], color=COLOR_RISING,     lw=1.0, label="Переход 0->1"),
        Line2D([0], [0], color=COLOR_FALLING,    lw=1.0, label="Переход 1->0"),
        Line2D([0], [0], color=COLOR_TRUE_BIT,   lw=0.7, ls=":", label="Истинные границы"),
    ]
    ax_sig.legend(handles=legend_items, fontsize=8, loc="upper right",
                  ncol=1, framealpha=0.85, edgecolor="lightgray")

    # ===================================================================
    # 4. Первая производная D1
    # ===================================================================
    ax_d1 = fig.add_subplot(gs[3])

    _add_roi_and_true_bits(ax_d1, det, sim_result.true_bit_centers)

    ax_d1.plot(xs, det.first_derivative, color=COLOR_D1, lw=1.4, zorder=5, label="D1")
    ax_d1.axhline(0, color="black", lw=0.8, zorder=4)
    ax_d1.axhline( det.peak_threshold, color=COLOR_THRESHOLD, lw=1.0, ls="--", alpha=0.8, label="±threshold")
    ax_d1.axhline(-det.peak_threshold, color=COLOR_THRESHOLD, lw=1.0, ls="--", alpha=0.8)

    # Вертикальные маркеры фронтов + точки на D1
    for edge in det.detected_edges:
        color = COLOR_RISING if edge.d1_value > 0 else COLOR_FALLING
        ax_d1.axvline(edge.position, color=color, lw=1.0, alpha=0.85, zorder=3)
        # Точка на уровне D1 в этой позиции
        ax_d1.plot(edge.position, edge.d1_value, "o", color=color, ms=4.5, zorder=6)

    ax_d1.set_xlim(-0.5, n_px - 0.5)
    ax_d1.set_xticks([])
    ax_d1.yaxis.set_tick_params(labelsize=7)
    ax_d1.set_ylabel("D1", fontsize=10)
    ax_d1.grid(axis="y", ls=":", lw=0.5, alpha=0.4)

    leg_d1 = [
        Line2D([0], [0], color=COLOR_D1,        lw=1.4, label=f"D1 (N={det.config.filter_order})"),
        Line2D([0], [0], color=COLOR_THRESHOLD, lw=1.0, ls="--", label="Предел нуля"),
        Line2D([0], [0], color=COLOR_RISING,    lw=1.0, label="Переход 0->1"),
        Line2D([0], [0], color=COLOR_FALLING,   lw=1.0, label="Переход 1->0"),
    ]
    ax_d1.legend(handles=leg_d1, fontsize=8, loc="upper right",
                 ncol=1, framealpha=0.85, edgecolor="lightgray")

    # ===================================================================
    # 5. Вторая производная D2 (нули = фронты)
    # ===================================================================
    ax_d2 = fig.add_subplot(gs[4])

    _add_roi_and_true_bits(ax_d2, det, sim_result.true_bit_centers)

    ax_d2.plot(xs, det.second_derivative, color=COLOR_D2, lw=1.4, zorder=5, label="D2")
    ax_d2.axhline(0, color="black", lw=0.8, zorder=4)

    # Нули D2 = обнаруженные фронты
    for edge in det.detected_edges:
        color = COLOR_RISING if edge.d1_value > 0 else COLOR_FALLING
        ax_d2.axvline(edge.position, color=color, lw=1.0, alpha=0.85, zorder=3)
        # Маркер нуля D2 на оси Y=0
        ax_d2.plot(edge.position, 0, "^", color=color, ms=5, zorder=6, clip_on=True)

    ax_d2.set_xlim(-0.5, n_px - 0.5)
    # Ось X с метками
    x_ticks = list(range(0, n_px + 1, 5))
    ax_d2.set_xticks(x_ticks)
    ax_d2.tick_params(axis="x", labelsize=7)
    ax_d2.set_xlabel("Pixel", fontsize=10)
    ax_d2.yaxis.set_tick_params(labelsize=7)
    ax_d2.set_ylabel("D2", fontsize=10)
    ax_d2.grid(axis="y", ls=":", lw=0.5, alpha=0.4)

    leg_d2 = [
        Line2D([0], [0], color=COLOR_D2,      lw=1.4, label="D2"),
        Line2D([0], [0], color=COLOR_RISING,  lw=1.0, label="Переход 0->1"),
        Line2D([0], [0], color=COLOR_FALLING, lw=1.0, label="Переход 1->0"),
        Line2D([0], [0], color=COLOR_ROI,     lw=1.0, ls="--", label="ROI"),
        Line2D([0], [0], color=COLOR_TRUE_BIT,lw=0.7, ls=":", label="Истинные биты"),
    ]
    ax_d2.legend(handles=leg_d2, fontsize=8, loc="upper right",
                 ncol=1, framealpha=0.85, edgecolor="lightgray")

    # Синхронизация осей X для панелей 2-5
    for ax in (ax_2d, ax_sig, ax_d1, ax_d2):
        ax.set_xlim(-0.5, n_px - 0.5)

    return fig


# ---------------------------------------------------------------------------
# Вывод подробных результатов в консоль (без изменений относительно оригинала)
# ---------------------------------------------------------------------------

def print_detailed_results(
    sim_result: SimulationResult,
    detection_result: EdgeDetectionResult,
) -> None:
    """Вывод подробных результатов в консоль."""
    det = detection_result
    config = sim_result.config

    print("\n" + "=" * 80)
    print("BLAIS-RIOUX EDGE DETECTOR — DETAILED RESULTS")
    print("=" * 80)

    print("\n--- SIMULATION PARAMETERS ---")
    print(f"  Bits: {config.n_bits}")
    print(f"  Bit width: {config.bit_width_px:.2f} px")
    print(f"  Blur sigma: {config.sigma_blur_px:.2f} px")
    print(f"  Noise sigma: {config.noise_sigma_adu:.1f} ADU")
    print(f"  Vignette: {config.vignette_strength:.2f}")
    print(f"  Pixels: {config.n_pixels}")
    print(f"  Seed: {config.seed}")

    print("\n--- PROCESSING PARAMETERS ---")
    print(f"  Filter order: N={det.config.filter_order}")
    print(f"  Peak threshold: {det.config.peak_threshold_rel * 100:.1f}%")
    print(
        f"  Min edge distance: {det.config.min_edge_distance_factor * 100:.1f}% of T")
    print(f"  Smoothing sigma: {det.config.smoothing_sigma:.2f} px")
    print(f"  MinMax window: {det.config.minmax_window_px} px")

    print("\n--- DETECTION RESULTS ---")
    print(f"  Detected edges: {len(det.detected_edges)}")
    print(f"  Bit segments: {len(det.bit_segments)}")
    print(f"  Recovered bits: {len(det.recovered_bits)}")
    print(f"  ROI: [{det.roi_start:.2f} - {det.roi_end:.2f}] px")
    print(f"  A priori bit width: {det.bit_width_px:.2f} px")
    print(f"  Measured period: {det.measured_bit_period:.3f} px")
    period_err = ((
                              det.measured_bit_period - det.bit_width_px) / det.bit_width_px * 100) if det.measured_bit_period > 0 else 0
    print(f"  Period deviation: {period_err:+.2f}%")
    print(f"  RMS edge error: {det.rms_edge_error:.4f} px")
    print(f"  Bit errors: {det.bit_errors}")
    print(f"  Accuracy: {det.accuracy:.1f}%")

    print("\n--- BIT SEGMENTS ---")
    print(
        f"  {'Idx':<4} {'Start':>8} {'End':>8} {'Dist':>7} {'nBits':>5} {'Period':>7} {'Bit':>4}")
    print(
        f"  {'-' * 4} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 5} {'-' * 7} {'-' * 4}")
    for i, seg in enumerate(det.bit_segments):
        print(
            f"  {i:<4} {seg.start_pos:>8.2f} {seg.end_pos:>8.2f} {seg.distance:>7.2f} {seg.n_bits:>5} {seg.measured_period:>7.3f} {seg.bit_value:>4}")

    print("\n--- RECOVERED BITS WITH POSITIONS ---")
    print(f"  {'Idx':<4} {'Position':>10} {'Value':>6} {'Segment':>8}")
    print(f"  {'-' * 4} {'-' * 10} {'-' * 6} {'-' * 8}")
    for i, bit in enumerate(det.recovered_bits):
        print(
            f"  {i:<4} {bit.position:>10.3f} {bit.value:>6} {bit.segment_idx:>8}")

    print("\n--- BIT SEQUENCES ---")
    print(f"  True bits:      {bits_to_string(sim_result.bits)}")
    print(f"  Recovered bits: {bits_to_string(det.recovered_bit_values)}")


def run_visualization(
    sim_config: SimulatorConfig,
    br_config: BRConfig,
    show_window: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, SimulationResult, EdgeDetectionResult]:
    """
    Запускает полный пайплайн: симуляция → обработка → визуализация.

    Parameters
    ----------
    sim_config : SimulatorConfig
        Конфигурация симуляции.
    br_config : BRConfig
        Конфигурация детектора.
    show_window : bool
        Показывать интерактивное окно matplotlib.
    save_path : Optional[str]
        Путь для сохранения изображения (PNG/PDF/SVG и т.д.).

    Returns
    -------
    Tuple[plt.Figure, SimulationResult, EdgeDetectionResult]
        Фигура, результат симуляции, результат детектирования.
    """
    sim_result       = simulate_ccd(sim_config)
    detection_result = detect_edges_and_recover_bits(
        sim_result.adc_signal,
        sim_result.bits,
        sim_result.true_edges,
        br_config,
    )

    print_detailed_results(sim_result, detection_result)

    fig = visualize_result(sim_result, detection_result)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nImage saved to: {save_path}")

    if show_window:
        plt.show()
    else:
        plt.close(fig)

    return fig, sim_result, detection_result


# ---------------------------------------------------------------------------
# Тестовый запуск
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim_config = SimulatorConfig(
        n_bits=24,
        bit_width_px=7.5,
        sigma_blur_px=1.5,
        noise_sigma_adu=60,
        vignette_strength=0.25,
        seed=7,
    )
    br_config = BRConfig(
        filter_order=4,
        peak_threshold_rel=0.15,
        min_edge_distance_factor=0.3,
        bit_width_px=7.5,
        smoothing_sigma=0.5,
        minmax_window_px=20,
    )
    run_visualization(sim_config, br_config)
