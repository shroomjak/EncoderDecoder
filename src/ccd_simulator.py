"""
ccd_simulator.py — Модуль генерации изображения ПЗС-линейки

Алгоритм генерации:
1. Случайная битовая последовательность
2. Непрерывный ступенчатый профиль на сверхдискретной сетке (oversample)
3. Гауссово размытие (PSF оптики)
4. Пикселизация с дисторсией сетки сэмплирования (division model)
5. Виньетирование (косинусное)
6. Квантование АЦП + гауссов шум

ИСПРАВЛЕНИЯ
-----------
BUG 1 (КРИТИЧНЫЙ) — Инверсия преобразования при симуляции дисторсии (шаг 4):
  Физика: пиксель сенсора с координатой x_d «смотрит» через дисторсионную оптику
  на точку идеального объекта с координатой x_u = undistort(x_d).
  Формула сэмплирования: profile_distorted[x_d] = profile_ideal[undistort(x_d)].
  Старый код вызывал distort_division_1d(), то есть применял ПРЯМОЕ преобразование
  вместо обратного → дисторсия имитировалась наоборот, коррекция только усугубляла.
  Исправление: заменена distort_division_1d() на локальную _undistort_norm()
  (формула 2r/(1+√(1−4kr²))).

BUG 2 — Удалена неиспользуемая функция distort_division() и импорт
  distort_division_1d из math1d (теперь не нужен).

BUG 3 (true_edges, x_u_start) — старый код вычислял x_u_start через
  `disc = max(0.0, 1 - 4k*x0²)` и `x0_u_norm = 2*x0 / (1 + disc**0.5)`,
  что является корректной формулой undistort. Оставлено без изменений,
  но теперь использует ту же вспомогательную функцию _undistort_norm() для
  единообразия и устойчивости.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.distortion.math1d import pixel_to_norm, norm_to_pixel


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

    distort_coeff: float = 0.1
    """Коэффициент дисторсии division model.
    k > 0 → бочкообразная (barrel): края объекта «сжимаются» к центру сенсора.
    k < 0 → подушкообразная (pincushion): края «растягиваются».
    """

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
    """Истинные позиции всех фронтов в M-point координатах undistorted сигнала
    (совпадает с пикселями ПЗС при k=0)."""

    true_bit_centers: np.ndarray
    """Истинные центры битовых ячеек в пикселях ПЗС."""

    config: SimulatorConfig
    """Конфигурация, использованная для симуляции."""


def _undistort_norm(x_d_norm: np.ndarray, k: float) -> np.ndarray:
    """
    Инверсная division model: x_d_norm → x_u_norm.

    Division model (прямая):  r_d = r_u / (1 + k * r_u²)
    Инверсная (undistort):    r_u = 2 * r_d / (1 + √(1 − 4k · r_d²))

    Описывает физику сенсора: пиксель с координатой x_d смотрит
    на точку идеального объекта с координатой x_u = undistort(x_d).
    """
    disc = np.clip(1.0 - 4.0 * k * x_d_norm ** 2, 0.0, None)
    return 2.0 * x_d_norm / (1.0 + np.sqrt(disc))


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
    eps = 0.05

    # 1. Случайные биты
    bits = rng.integers(0, 2, config.n_bits)

    # 2. Сверхдискретная сетка
    total_sub = config.n_pixels * config.oversample
    x_sub = np.arange(total_sub, dtype=np.float64) / config.oversample

    bit_span = config.n_bits * config.bit_width_px
    x_start = (config.n_pixels - bit_span) / 2.0

    # Ступенчатый профиль
    profile = np.zeros(total_sub, dtype=np.float64)
    raw_true_edges = []

    for i, b in enumerate(bits):
        shift = np.random.uniform(-eps, eps)
        lo = x_start + i * config.bit_width_px + shift
        hi = lo + config.bit_width_px  # shift не прибавляется к ширине ячейки

        profile[(x_sub >= lo) & (x_sub < hi)] = float(b)
        # Коррекция -0.5: пиксель i представляет интеграл [i, i+1),
        # его центр в позиции i+0.5. Но signal[i] интерпретируется как позиция i.
        raw_true_edges.append(lo - 0.5)

    shift = np.random.uniform(-eps, eps)
    end = x_start + config.n_bits * config.bit_width_px + shift
    raw_true_edges.append(end - 0.5)

    # true_edges задаются в undistorted (идеальных) пикселях ПЗС.
    # recover_bits ожидает координаты в M-point пространстве undistorted сигнала,
    # которое blais_rioux строит как: x_u_new = x_u_px[0] + arange(M), step=1.
    # x_u_px[0] = undistort(x_d=0) — смещение нуля undistorted сетки.
    # Для k=0: смещение = 0, M=N, M-point совпадает с px.
    if abs(config.distort_coeff) > 1e-15:
        c = 0.5 * (config.n_pixels - 1)
        s = max(c, 1.0)
        x0_norm = (0.0 - c) / s  # x_d=0 нормированный
        x0_u_norm = float(_undistort_norm(np.array([x0_norm]), config.distort_coeff)[0])
        x_u_start = x0_u_norm * s + c  # в пикселях
        true_edges = np.array(raw_true_edges, dtype=np.float64) - x_u_start
    else:
        true_edges = np.array(raw_true_edges, dtype=np.float64)
    true_bit_centers = (true_edges[:-1] + true_edges[1:]) / 2.0

    # 3. Гауссово размытие (в недисторсированном пространстве)
    sigma_sub = config.sigma_blur_px * config.oversample
    profile_blur = gaussian_filter1d(profile, sigma=sigma_sub)

    # 4. Пикселизация с применением дисторсии к сетке сэмплирования.
    #
    # Физика: каждый пиксель сенсора с координатой x_d «смотрит» через
    # дисторсионную оптику на точку идеального объекта с координатой
    #   x_u = undistort(x_d)
    # Формула сэмплирования:
    #   profile_distorted[x_d] = profile_ideal[undistort(x_d)]
    #
    # ИСПРАВЛЕНО: старый код применял distort() вместо undistort().
    # Это приводило к тому, что дисторсия вносилась в обратную сторону:
    # «сжатые» биты растягивались и наоборот. После коррекции в blais_rioux
    # ошибка накапливалась вместо устранения.
    if abs(config.distort_coeff) > 1e-15:
        x_sub_norm = pixel_to_norm(x_sub, config.n_pixels)
        # FIX: undistort — инверсная модель (было: distort_division_1d)
        x_undist_norm = _undistort_norm(x_sub_norm, config.distort_coeff)
        x_sub_undist = norm_to_pixel(x_undist_norm, config.n_pixels)
        # Сэмплируем идеальный профиль по undistorted-позициям
        profile_sampled = np.interp(x_sub_undist, x_sub, profile_blur)
    else:
        profile_sampled = profile_blur

    pixel_vals = profile_sampled.reshape(config.n_pixels, config.oversample).mean(axis=1)

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
        config=config,
    )


if __name__ == "__main__":
    # Тестовый запуск
    result = simulate_ccd()
    print(f"Bits: {result.bits}")
    print(f"ADC signal shape: {result.adc_signal.shape}")
    print(f"True edges: {result.true_edges[:5]}...")
