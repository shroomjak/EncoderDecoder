"""
ccd_simulator.py — Модуль генерации изображения ПЗС-линейки

Алгоритм генерации:
1. Случайная битовая последовательность
2. Непрерывный ступенчатый профиль на сверхдискретной сетке (oversample)
3. Гауссово размытие (PSF оптики)
4. Пикселизация: усреднение (интегрирование) по ячейкам ПЗС
5. Виньетирование (косинусное)
6. Квантование АЦП + гауссов шум
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class SimulatorConfig:
    """Конфигурация симулятора ПЗС-линейки."""
    
    n_bits: int = 24
    """Количество бит в последовательности."""
    
    bit_width_px: float = 7.5
    """Ширина одного бита в пикселях ПЗС (дробная допустима)."""
    
    sigma_blur_px: float = 0.5
    """Стандартное отклонение гауссова размытия в пикселях ПЗС."""
    
    oversample: int = 32
    """Число отсчётов сверхдискретной сетки на один пиксель ПЗС."""
    
    n_pixels: int = 128
    """Число пикселей ПЗС-линейки."""
    
    adc_bits: int = 12
    """Разрядность АЦП."""
    
    noise_sigma_adu: float = 60.0
    """СКО гауссова шума в квантах АЦП."""
    
    vignette_strength: float = 0.25
    """Сила виньетирования: 0 — нет, 1 — максимальное (центр=1, края=0)."""
    
    seed: Optional[int] = 7
    """Начальное значение генератора случайных чисел (None — случайный)."""


@dataclass
class SimulationResult:
    """Результат симуляции ПЗС-линейки."""
    
    bits: np.ndarray
    """Истинная битовая последовательность [n_bits]."""
    
    adc_signal: np.ndarray
    """Целочисленный сигнал АЦП [n_pixels], dtype=int32."""
    
    true_edges: np.ndarray
    """Истинные позиции ВСЕХ фронтов (включая внешние) в пикселях ПЗС."""
    
    true_bit_centers: np.ndarray
    """Истинные центры битовых ячеек в пикселях ПЗС."""
    
    profile_continuous: np.ndarray
    """Непрерывный профиль после размытия (на сверхдискретной сетке)."""
    
    x_continuous: np.ndarray
    """Координата сверхдискретной сетки в пикселях ПЗС."""
    
    config: SimulatorConfig
    """Конфигурация, использованная для симуляции."""


def simulate_ccd(config: Optional[SimulatorConfig] = None) -> SimulationResult:
    """
    Генерирует изображение случайной битовой последовательности на ПЗС-линейке.

    Parameters
    ----------
    config : SimulatorConfig, optional
        Параметры симуляции. Если None — используются значения по умолчанию.

    Returns
    -------
    SimulationResult
        Структура с сигналом АЦП, истинными позициями фронтов и центров бит.
    """
    if config is None:
        config = SimulatorConfig()

    rng = np.random.default_rng(config.seed)
    adc_max = (1 << config.adc_bits) - 1

    # 1. Случайные биты
    bits = rng.integers(0, 2, config.n_bits)

    # 2. Сверхдискретная сетка
    total_sub = config.n_pixels * config.oversample
    x_sub = np.arange(total_sub, dtype=np.float64) / config.oversample

    bit_span = config.n_bits * config.bit_width_px
    x_start = (config.n_pixels - bit_span) / 2.0

    # Ступенчатый профиль
    profile = np.zeros(total_sub, dtype=np.float64)
    true_edges = []
    
    for i, b in enumerate(bits):
        lo = x_start + i * config.bit_width_px
        hi = lo + config.bit_width_px
        profile[(x_sub >= lo) & (x_sub < hi)] = float(b)
        # Коррекция -0.5: пиксель i представляет интеграл [i, i+1),
        # его центр в позиции i+0.5. Но signal[i] интерпретируется как позиция i.
        true_edges.append(lo - 0.5)
    
    true_edges.append(x_start + config.n_bits * config.bit_width_px - 0.5)
    true_edges = np.array(true_edges)
    true_bit_centers = (true_edges[:-1] + true_edges[1:]) / 2.0

    # 3. Гауссово размытие
    sigma_sub = config.sigma_blur_px * config.oversample
    profile_blur = gaussian_filter1d(profile, sigma=sigma_sub)

    # 4. Пикселизация (усреднение по ячейкам)
    pixel_vals = profile_blur.reshape(config.n_pixels, config.oversample).mean(axis=1)

    # Нормировка [0, 1]
    pv_min, pv_max = pixel_vals.min(), pixel_vals.max()
    pixel_vals = (pixel_vals - pv_min) / (pv_max - pv_min + 1e-12)

    # 5. Виньетирование
    idx_v = np.linspace(-1.0, 1.0, config.n_pixels)
    vignette = 1.0 - config.vignette_strength * (1.0 - np.cos(np.pi * idx_v / 2.0))
    pixel_vals *= vignette

    # 6. АЦП + шум
    adc_ideal = pixel_vals * adc_max
    adc_noisy = adc_ideal + rng.normal(0.0, config.noise_sigma_adu, config.n_pixels)
    adc_noisy = np.clip(np.round(adc_noisy), 0, adc_max).astype(np.int32)

    return SimulationResult(
        bits=bits,
        adc_signal=adc_noisy,
        true_edges=true_edges,
        true_bit_centers=true_bit_centers,
        profile_continuous=profile_blur,
        x_continuous=x_sub,
        config=config,
    )


if __name__ == "__main__":
    # Тестовый запуск
    result = simulate_ccd()
    print(f"Bits: {result.bits}")
    print(f"ADC signal shape: {result.adc_signal.shape}")
    print(f"True edges: {result.true_edges[:5]}...")
