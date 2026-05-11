#!/usr/bin/env python3
"""
main.py — CLI интерфейс для Blais-Rioux Edge Detector

Использование:
    python main.py --help
    python main.py --n-bits 24 --bit-width 7.5 --noise 60 --seed 42
    python main.py --filter-order 2 --threshold 20 --save output.png
"""

import click
from typing import Optional

from src.ccd_simulator import SimulatorConfig
from src.blais_rioux import BlaisRiouxConfig
from src.visualizer import run_visualization


@click.command()
@click.option('--n-bits', default=24, type=int, help='Количество бит в последовательности')
@click.option('--bit-width', default=5, type=float, help='Ширина бита в пикселях (априори)')
@click.option('--blur', default=0.5, type=float, help='Сигма гауссова размытия оптики')
@click.option('--noise', default=100.0, type=float, help='Сигма шума АЦП (ADU)')
@click.option('--vignette', default=0.5, type=float, help='Сила виньетирования [0..1]')
@click.option('--n-pixels', default=200, type=int, help='Количество пикселей ПЗС')
@click.option('--seed', default=7, type=int, help='Seed для генератора случайных чисел')
@click.option('--filter-order', default=2, type=click.Choice(['2', '4']), help='Порядок КИХ фильтра')
@click.option('--threshold', default=15, type=float, help='Порог |D1| в %% от максимума')
@click.option('--min-dist', default=30, type=float, help='Мин. расстояние между фронтами в %% от T')
@click.option('--smoothing', default=0.2, type=float, help='Сигма сглаживания перед D1')
@click.option('--minmax-window', default=50, type=int, help='Ширина окна min/max нормировки')
@click.option('--save', default=None, type=str, help='Путь для сохранения изображения')
@click.option('--no-window', is_flag=True, help='Не показывать окно (только консольный вывод)')
def main(
    n_bits: int,
    bit_width: float,
    blur: float,
    noise: float,
    vignette: float,
    n_pixels: int,
    seed: int,
    filter_order: str,
    threshold: float,
    min_dist: float,
    smoothing: float,
    minmax_window: int,
    save: Optional[str],
    no_window: bool
) -> None:
    """
    Blais-Rioux Edge Detector — субпиксельное детектирование фронтов на ПЗС-линейке.
    
    Генерирует симуляцию сигнала ПЗС, применяет алгоритм Blais-Rioux для
    детектирования фронтов и восстановления битовой последовательности.
    
    \b
    Примеры:
        python main.py
        python main.py --n-bits 32 --noise 100 --seed 123
        python main.py --filter-order 2 --threshold 25
        python main.py --save result.png --no-window
    """
    # Конфигурация симуляции
    sim_config = SimulatorConfig(
        n_bits=n_bits,
        bit_width_px=bit_width,
        sigma_blur_px=blur,
        noise_sigma_adu=noise,
        vignette_strength=vignette,
        n_pixels=n_pixels,
        seed=seed
    )
    
    # Конфигурация обработки
    br_config = BlaisRiouxConfig(
        filter_order=int(filter_order),
        peak_threshold_rel=threshold / 100.0,
        min_edge_distance_factor=min_dist / 100.0,
        bit_width_px=bit_width,
        smoothing_sigma=smoothing,
        minmax_window_px=minmax_window
    )
    
    # Запуск
    run_visualization(
        sim_config=sim_config,
        br_config=br_config,
        show_window=not no_window,
        save_path=save
    )


if __name__ == '__main__':
    main()
