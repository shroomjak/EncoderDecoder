"""
CLI интерфейс для BR Edge Detector
"""

import click
from typing import Optional

from src.ccd_simulator import SimulatorConfig
from src.blais_rioux import BRConfig
from src.visualizer import run_visualization


@click.command()
@click.option('--n-bits', default=24, type=int, help='Количество бит в последовательности')
@click.option('--bit-width', default=5, type=float, help='Ширина бита в пикселях (априори)')
@click.option('--blur', default=0.5, type=float, help='Размытие оптики')
@click.option('--noise', default=100.0, type=float, help='Шум АЦП (ADU)')
@click.option('--vignette', default=0.5, type=float, help='Сила виньетирования [0..1]')
@click.option('--distort', default=0.1, type=float, help='Сила дисторсии [-1..1]')
@click.option('--n-pixels', default=200, type=int, help='Количество пикселей ПЗС')
@click.option('--seed', default=7, type=int, help='Seed для генератора случайных чисел')
@click.option('--filter-order', default=2, type=click.Choice(['2', '4']), help='Порядок КИХ фильтра')
@click.option('--threshold', default=15, type=float, help='Порог |D1| в %% от максимума')
@click.option('--min-dist', default=30, type=float, help='Мин. расстояние между фронтами в %% от T')
@click.option('--smoothing', default=0.2, type=float, help='Cглаживание перед D1')
@click.option('--minmax-window', default=50, type=int, help='Ширина окна min-max нормировки')
@click.option('--save', default=None, type=str, help='Путь для сохранения изображения')
@click.option('--no-window', is_flag=True, help='Не показывать окно (только консольный вывод)')
def main(
    n_bits: int,
    bit_width: float,
    blur: float,
    noise: float,
    vignette: float,
    distort: float,
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
    Субпиксельное детектирование фронтов на сигнале с ПЗС-линейки.
    
    Генерирует симуляцию сигнала ПЗС, применяет алгоритм BR-N для
    детектирования фронтов и восстановления битовой последовательности.
    """

    # Конфигурация симуляции
    sim_config = SimulatorConfig(
        n_bits=n_bits,
        bit_width_px=bit_width,
        sigma_blur_px=blur,
        noise_sigma_adu=noise,
        vignette_strength=vignette,
        distort_coeff=distort,
        n_pixels=n_pixels,
        seed=seed
    )
    
    # Конфигурация обработки
    br_config = BRConfig(
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
