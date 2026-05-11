"""
visualizer.py — Модуль визуализации через OpenCV

Создаёт окно с четырьмя графиками друг под другом:
1. Исходный сигнал как псевдо-2D изображение (оттенки серого)
2. Размеченный сглаженный нормированный сигнал с восстановленными битами
3. График первой производной D1 с разметкой фронтов
4. График второй производной D2

Над графиками выводится диагностическая информация.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from src.ccd_simulator import SimulatorConfig, SimulationResult, simulate_ccd
from src.blais_rioux import (
    BlaisRiouxConfig, EdgeDetectionResult, DetectedEdge,
    detect_edges_and_recover_bits
)


# Строгая классическая гамма, BGR для OpenCV
COLOR_BG = (255, 255, 255)         # white
COLOR_TEXT = (32, 32, 32)          # near-black
COLOR_TEXT_DIM = (110, 110, 110)   # medium gray
COLOR_GRID = (200, 200, 200)       # light gray

COLOR_SIGNAL = (180, 119, 31)      # tab:blue
COLOR_SMOOTHED = (14, 127, 255)    # tab:orange
COLOR_LOCAL_NORM = (44, 160, 44)   # tab:green
COLOR_RECOVERED = (40, 39, 214)    # tab:red

COLOR_D1 = (189, 103, 148)         # tab:purple
COLOR_D2 = (75, 86, 140)           # tab:brown

COLOR_TRUE_EDGE = (34, 189, 188)   # tab:olive
COLOR_EDGE_RISING = (207, 190, 23) # tab:cyan
COLOR_EDGE_FALLING = (194, 119, 227) # tab:pink

COLOR_ROI = (127, 127, 127)        # tab:gray
COLOR_THRESHOLD = (14, 127, 255)   # same orange as smoothed
COLOR_ERROR = (40, 39, 214)        # same red as recovered
COLOR_CORRECT = (44, 160, 44)      # same green as local_norm


def draw_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int] = COLOR_TEXT,
    scale: float = 0.4,
    thickness: int = 1
) -> None:
    """Отрисовка текста на изображении."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_line(
    img: np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    color: Tuple[int, int, int],
    thickness: int = 1
) -> None:
    """Отрисовка линии."""
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def create_signal_strip(
    signal: np.ndarray,
    width: int,
    height: int = 100
) -> np.ndarray:
    """
    Создаёт псевдо-2D изображение сигнала (растянутое по вертикали).
    
    Parameters
    ----------
    signal : np.ndarray
        Нормализованный сигнал [0, 1].
    width : int
        Ширина выходного изображения.
    height : int
        Высота полосы.
        
    Returns
    -------
    np.ndarray
        Изображение в оттенках серого (BGR).
    """
    n = len(signal)
    
    # Масштабирование по ширине
    scale_x = width / n
    
    # Создаём изображение
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Нормализуем сигнал к [0, 255]
    s_min, s_max = signal.min(), signal.max()
    signal_norm = (signal - s_min) / (s_max - s_min + 1e-12)
    signal_uint8 = (signal_norm * 255).astype(np.uint8)
    
    # Заполняем каждый столбец
    for i in range(n):
        x_start = int(i * scale_x)
        x_end = int((i + 1) * scale_x)
        gray = signal_uint8[i]
        img[:, x_start:x_end] = (gray, gray, gray)
    
    return img


def draw_graph(
    img: np.ndarray,
    signal: np.ndarray,
    y_offset: int,
    height: int,
    width: int,
    color: Tuple[int, int, int],
    y_range: Optional[Tuple[float, float]] = None,
    line_width: int = 1
) -> Tuple[float, float]:
    """
    Отрисовка графика сигнала.
    
    Parameters
    ----------
    img : np.ndarray
        Изображение для отрисовки.
    signal : np.ndarray
        Сигнал для отображения.
    y_offset : int
        Смещение по Y (верх графика).
    height : int
        Высота области графика.
    width : int
        Ширина области графика.
    color : Tuple[int, int, int]
        Цвет линии.
    y_range : Optional[Tuple[float, float]]
        Диапазон по Y. Если None — автоматический.
    line_width : int
        Толщина линии.
        
    Returns
    -------
    Tuple[float, float]
        Использованный диапазон (y_min, y_max).
    """
    n = len(signal)
    scale_x = width / n
    
    if y_range is None:
        y_min, y_max = signal.min(), signal.max()
        # Добавляем отступ
        margin = (y_max - y_min) * 0.1 + 1e-12
        y_min -= margin
        y_max += margin
    else:
        y_min, y_max = y_range
    
    def to_screen_y(val: float) -> int:
        normalized = (val - y_min) / (y_max - y_min + 1e-12)
        return int(y_offset + height - normalized * height)
    
    # Отрисовка линии
    prev_x, prev_y = None, None
    for i in range(n):
        x = int(i * scale_x + scale_x / 2)
        y = to_screen_y(signal[i])
        y = max(y_offset, min(y_offset + height - 1, y))
        
        if prev_x is not None:
            draw_line(img, prev_x, prev_y, x, y, color, line_width)
        
        prev_x, prev_y = x, y
    
    return y_min, y_max


def draw_vertical_line(
    img: np.ndarray,
    x_pos: float,
    y_offset: int,
    height: int,
    width: int,
    n_pixels: int,
    color: Tuple[int, int, int],
    dashed: bool = False,
    thickness: int = 1
) -> None:
    """
    Отрисовка вертикальной линии (фронт, ROI и т.д.).
    
    Parameters
    ----------
    x_pos : float
        Позиция в координатах сигнала.
    """
    scale_x = width / n_pixels
    screen_x = int(x_pos * scale_x + scale_x / 2)
    
    if screen_x < 0 or screen_x >= width:
        return
    
    if dashed:
        # Пунктирная линия
        for y in range(y_offset, y_offset + height, 6):
            y_end = min(y + 3, y_offset + height)
            draw_line(img, screen_x, y, screen_x, y_end, color, thickness)
    else:
        draw_line(img, screen_x, y_offset, screen_x, y_offset + height, color, thickness)


def draw_horizontal_line(
    img: np.ndarray,
    y_val: float,
    y_offset: int,
    height: int,
    width: int,
    y_min: float,
    y_max: float,
    color: Tuple[int, int, int],
    dashed: bool = True
) -> None:
    """Отрисовка горизонтальной линии (порог и т.д.)."""
    normalized = (y_val - y_min) / (y_max - y_min + 1e-12)
    screen_y = int(y_offset + height - normalized * height)
    
    if screen_y < y_offset or screen_y >= y_offset + height:
        return
    
    if dashed:
        for x in range(0, width, 8):
            x_end = min(x + 4, width)
            draw_line(img, x, screen_y, x_end, screen_y, color, 1)
    else:
        draw_line(img, 0, screen_y, width, screen_y, color, 1)


def bits_to_string(bits: np.ndarray) -> str:
    """Преобразование массива бит в строку."""
    return ''.join(str(int(b)) for b in bits)


def visualize_result(
    sim_result: SimulationResult,
    detection_result: EdgeDetectionResult,
    window_name: str = "Blais-Rioux Edge Detector"
) -> np.ndarray:
    """
    Создаёт визуализацию результатов обработки.
    
    Parameters
    ----------
    sim_result : SimulationResult
        Результат симуляции.
    detection_result : EdgeDetectionResult
        Результат детектирования.
    window_name : str
        Имя окна.
        
    Returns
    -------
    np.ndarray
        Изображение визуализации.
    """
    config = sim_result.config
    det = detection_result
    n_pixels = config.n_pixels
    
    # Размеры
    width = max(800, n_pixels * 3)
    header_height = 120
    strip_height = 40
    graph_height = 150
    margin = 10
    
    total_height = header_height + strip_height + 3 * (graph_height + margin) + margin
    
    # Создаём изображение
    img = np.full((total_height, width, 3), COLOR_BG, dtype=np.uint8)
    
    # === HEADER: Диагностическая информация ===
    y = 20
    
    # Строка 1: Параметры симуляции
    params_str = (
        f"Bits: {config.n_bits}  |  "
        f"BitWidth: {config.bit_width_px:.1f}px  |  "
        f"Blur: {config.sigma_blur_px:.1f}px  |  "
        f"Noise: {config.noise_sigma_adu:.0f} ADU  |  "
        f"Vignette: {config.vignette_strength:.2f}  |  "
        f"Seed: {config.seed}"
    )
    draw_text(img, params_str, (10, y), COLOR_TEXT_DIM, 0.35)
    
    y += 18
    # Строка 2: Параметры обработки
    proc_str = (
        f"Filter N={det.config.filter_order}  |  "
        f"Threshold: {det.config.peak_threshold_rel*100:.0f}%  |  "
        f"MinDist: {det.config.min_edge_distance_factor*100:.0f}%T  |  "
        f"Smooth: {det.config.smoothing_sigma:.1f}px  |  "
        f"MinMax: {det.config.minmax_window_px}px"
    )
    draw_text(img, proc_str, (10, y), COLOR_TEXT_DIM, 0.35)
    
    y += 22
    # Строка 3: Результаты
    period_err = ((det.measured_bit_period - det.bit_width_px) / det.bit_width_px * 100) if det.measured_bit_period > 0 else 0
    result_str = (
        f"Accuracy: {det.accuracy:.1f}%  |  "
        f"Edges: {len(det.detected_edges)}  |  "
        f"RMS: {det.rms_edge_error:.3f}px  |  "
        f"Period: {det.measured_bit_period:.2f}px ({period_err:+.2f}%)  |  "
        f"ROI: [{det.roi_start:.1f} - {det.roi_end:.1f}]"
    )
    color_acc = COLOR_CORRECT if det.accuracy >= 99 else (COLOR_THRESHOLD if det.accuracy >= 90 else COLOR_ERROR)
    draw_text(img, result_str, (10, y), color_acc, 0.4)
    
    y += 22
    # Строка 4: Истинные биты
    true_bits_str = f"True:      {bits_to_string(sim_result.bits)}"
    draw_text(img, true_bits_str, (10, y), COLOR_TRUE_EDGE, 0.35)
    
    y += 16
    # Строка 5: Восстановленные биты
    rec_bits_str = f"Recovered: {bits_to_string(det.recovered_bit_values)}"
    draw_text(img, rec_bits_str, (10, y), COLOR_RECOVERED, 0.35)
    
    # === PANEL 1: Исходный сигнал как 2D полоса ===
    panel1_y = header_height
    
    # Нормализация сигнала
    signal_norm = (sim_result.adc_signal - sim_result.adc_signal.min()) / (sim_result.adc_signal.max() - sim_result.adc_signal.min() + 1e-12)
    strip_img = create_signal_strip(signal_norm, width, strip_height)
    img[panel1_y:panel1_y + strip_height, :] = strip_img
    
    # Подпись
    draw_text(img, "Raw CCD Signal", (5, panel1_y + 12), COLOR_TEXT_DIM, 0.35)
    
    # === PANEL 2: Сглаженный и нормированный сигнал ===
    panel2_y = panel1_y + strip_height + margin
    
    # Фон
    cv2.rectangle(img, (0, panel2_y), (width, panel2_y + graph_height), COLOR_BG, -1)
    
    # Сетка
    for i in range(5):
        y_grid = panel2_y + int(i * graph_height / 4)
        draw_line(img, 0, y_grid, width, y_grid, COLOR_GRID, 1)
    
    # ROI
    draw_vertical_line(img, det.roi_start, panel2_y, graph_height, width, n_pixels, COLOR_ROI, dashed=True)
    draw_vertical_line(img, det.roi_end, panel2_y, graph_height, width, n_pixels, COLOR_ROI, dashed=True)
    
    # Сглаженный сигнал
    draw_graph(img, det.smoothed_signal, panel2_y, graph_height, width, COLOR_SMOOTHED, y_range=(0, 1))
    
    # Локально нормированный сигнал
    draw_graph(img, det.local_norm_signal, panel2_y, graph_height, width, COLOR_LOCAL_NORM, y_range=(0, 1))
    
    # Восстановленные битовые уровни
    scale_x = width / n_pixels
    for seg in det.bit_segments:
        x1 = int(seg.start_pos * scale_x)
        x2 = int(seg.end_pos * scale_x)
        y_level = panel2_y + graph_height - int(seg.bit_value * graph_height * 0.8) - int(graph_height * 0.1)
        draw_line(img, x1, y_level, x2, y_level, COLOR_RECOVERED, 2)
    
    # Фронты
    for edge in det.detected_edges:
        color = COLOR_EDGE_RISING if edge.d1_value > 0 else COLOR_EDGE_FALLING
        draw_vertical_line(img, edge.position, panel2_y, graph_height, width, n_pixels, color, thickness=1)
    
    draw_text(img, "Smoothed + LocalNorm + Recovered Bits", (5, panel2_y + 12), COLOR_TEXT_DIM, 0.35)
    
    # === PANEL 3: Первая производная D1 ===
    panel3_y = panel2_y + graph_height + margin
    
    cv2.rectangle(img, (0, panel3_y), (width, panel3_y + graph_height), COLOR_BG, -1)
    
    # Сетка
    for i in range(5):
        y_grid = panel3_y + int(i * graph_height / 4)
        draw_line(img, 0, y_grid, width, y_grid, COLOR_GRID, 1)
    
    # ROI
    draw_vertical_line(img, det.roi_start, panel3_y, graph_height, width, n_pixels, COLOR_ROI, dashed=True)
    draw_vertical_line(img, det.roi_end, panel3_y, graph_height, width, n_pixels, COLOR_ROI, dashed=True)
    
    # D1
    y_min, y_max = draw_graph(img, det.first_derivative, panel3_y, graph_height, width, COLOR_D1)
    
    # Нулевая линия
    draw_horizontal_line(img, 0, panel3_y, graph_height, width, y_min, y_max, COLOR_GRID, dashed=True)
    
    # Порог
    draw_horizontal_line(img, det.peak_threshold, panel3_y, graph_height, width, y_min, y_max, COLOR_THRESHOLD, dashed=True)
    draw_horizontal_line(img, -det.peak_threshold, panel3_y, graph_height, width, y_min, y_max, COLOR_THRESHOLD, dashed=True)
    
    # Фронты
    for edge in det.detected_edges:
        color = COLOR_EDGE_RISING if edge.d1_value > 0 else COLOR_EDGE_FALLING
        draw_vertical_line(img, edge.position, panel3_y, graph_height, width, n_pixels, color, thickness=1)
    
    draw_text(img, f"D1 (Blais-Rioux N={det.config.filter_order})", (5, panel3_y + 12), COLOR_TEXT_DIM, 0.35)
    
    # === PANEL 4: Вторая производная D2 ===
    panel4_y = panel3_y + graph_height + margin
    
    cv2.rectangle(img, (0, panel4_y), (width, panel4_y + graph_height), COLOR_BG, -1)
    
    # Сетка
    for i in range(5):
        y_grid = panel4_y + int(i * graph_height / 4)
        draw_line(img, 0, y_grid, width, y_grid, COLOR_GRID, 1)
    
    # ROI
    draw_vertical_line(img, det.roi_start, panel4_y, graph_height, width, n_pixels, COLOR_ROI, dashed=True)
    draw_vertical_line(img, det.roi_end, panel4_y, graph_height, width, n_pixels, COLOR_ROI, dashed=True)
    
    # D2
    y_min, y_max = draw_graph(img, det.second_derivative, panel4_y, graph_height, width, COLOR_D2)
    
    # Нулевая линия
    draw_horizontal_line(img, 0, panel4_y, graph_height, width, y_min, y_max, COLOR_GRID, dashed=True)
    
    # Фронты (нули D2)
    for edge in det.detected_edges:
        color = COLOR_EDGE_RISING if edge.d1_value > 0 else COLOR_EDGE_FALLING
        draw_vertical_line(img, edge.position, panel4_y, graph_height, width, n_pixels, color, thickness=1)
    
    draw_text(img, "D2 (zeros = edges)", (5, panel4_y + 12), COLOR_TEXT_DIM, 0.35)
    
    return img


def print_detailed_results(
    sim_result: SimulationResult,
    detection_result: EdgeDetectionResult
) -> None:
    """
    Выводит детальную информацию в консоль.
    
    Parameters
    ----------
    sim_result : SimulationResult
        Результат симуляции.
    detection_result : EdgeDetectionResult
        Результат детектирования.
    """
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
    print(f"  Peak threshold: {det.config.peak_threshold_rel*100:.1f}%")
    print(f"  Min edge distance: {det.config.min_edge_distance_factor*100:.1f}% of T")
    print(f"  Smoothing sigma: {det.config.smoothing_sigma:.2f} px")
    print(f"  MinMax window: {det.config.minmax_window_px} px")
    
    print("\n--- DETECTION RESULTS ---")
    print(f"  Detected edges: {len(det.detected_edges)}")
    print(f"  Bit segments: {len(det.bit_segments)}")
    print(f"  Recovered bits: {len(det.recovered_bits)}")
    print(f"  ROI: [{det.roi_start:.2f} - {det.roi_end:.2f}] px")
    print(f"  A priori bit width: {det.bit_width_px:.2f} px")
    print(f"  Measured period: {det.measured_bit_period:.3f} px")
    period_err = ((det.measured_bit_period - det.bit_width_px) / det.bit_width_px * 100) if det.measured_bit_period > 0 else 0
    print(f"  Period deviation: {period_err:+.2f}%")
    print(f"  RMS edge error: {det.rms_edge_error:.4f} px")
    print(f"  Bit errors: {det.bit_errors}")
    print(f"  Accuracy: {det.accuracy:.1f}%")
    
    print("\n--- BIT SEGMENTS ---")
    print(f"  {'Idx':<4} {'Start':>8} {'End':>8} {'Dist':>7} {'nBits':>5} {'Period':>7} {'Bit':>4}")
    print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*7} {'-'*5} {'-'*7} {'-'*4}")
    for i, seg in enumerate(det.bit_segments):
        print(f"  {i:<4} {seg.start_pos:>8.2f} {seg.end_pos:>8.2f} {seg.distance:>7.2f} {seg.n_bits:>5} {seg.measured_period:>7.3f} {seg.bit_value:>4}")
    
    print("\n--- RECOVERED BITS WITH POSITIONS ---")
    print(f"  {'Idx':<4} {'Position':>10} {'Value':>6} {'Segment':>8}")
    print(f"  {'-'*4} {'-'*10} {'-'*6} {'-'*8}")
    for i, bit in enumerate(det.recovered_bits):
        print(f"  {i:<4} {bit.position:>10.3f} {bit.value:>6} {bit.segment_idx:>8}")
    
    print("\n--- BIT SEQUENCES ---")
    print(f"  True bits:      {bits_to_string(sim_result.bits)}")
    print(f"  Recovered bits: {bits_to_string(det.recovered_bit_values)}")
    print(f"  Aligned true:   {bits_to_string(det.aligned_true_bits)}")
    print(f"  Aligned recov:  {bits_to_string(det.aligned_recovered_bits)}")
    
    # Сравнение
    print("\n--- COMPARISON (aligned) ---")
    errors_str = ""
    for i in range(len(det.aligned_true_bits)):
        if i < len(det.aligned_recovered_bits):
            if det.aligned_true_bits[i] == det.aligned_recovered_bits[i]:
                errors_str += "."
            else:
                errors_str += "X"
        else:
            errors_str += "?"
    print(f"  Errors:         {errors_str}")
    
    print("\n" + "=" * 80)


def run_visualization(
    sim_config: SimulatorConfig,
    br_config: BlaisRiouxConfig,
    show_window: bool = True,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, SimulationResult, EdgeDetectionResult]:
    """
    Запускает полный пайплайн: симуляция → обработка → визуализация.
    
    Parameters
    ----------
    sim_config : SimulatorConfig
        Конфигурация симуляции.
    br_config : BlaisRiouxConfig
        Конфигурация детектора.
    show_window : bool
        Показывать окно.
    save_path : Optional[str]
        Путь для сохранения изображения.
        
    Returns
    -------
    Tuple[np.ndarray, SimulationResult, EdgeDetectionResult]
        Изображение, результат симуляции, результат детектирования.
    """
    # Симуляция
    sim_result = simulate_ccd(sim_config)
    
    # Обработка
    detection_result = detect_edges_and_recover_bits(
        sim_result.adc_signal,
        sim_result.bits,
        sim_result.true_edges,
        br_config
    )
    
    # Вывод в консоль
    print_detailed_results(sim_result, detection_result)
    
    # Визуализация
    img = visualize_result(sim_result, detection_result)
    
    # Сохранение
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"\nImage saved to: {save_path}")
    
    # Показ окна
    if show_window:
        cv2.imshow("Blais-Rioux Edge Detector", img)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img, sim_result, detection_result


if __name__ == "__main__":
    # Тестовый запуск
    sim_config = SimulatorConfig(
        n_bits=24,
        bit_width_px=7.5,
        sigma_blur_px=1.5,
        noise_sigma_adu=60,
        vignette_strength=0.25,
        seed=7
    )
    
    br_config = BlaisRiouxConfig(
        filter_order=4,
        peak_threshold_rel=0.15,
        min_edge_distance_factor=0.3,
        bit_width_px=7.5,
        smoothing_sigma=0.5,
        minmax_window_px=20
    )
    
    run_visualization(sim_config, br_config)
