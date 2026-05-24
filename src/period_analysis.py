"""
Анализ периодического пилообразного сигнала абсолютного энкодера.

Алгоритм:
  1. Чтение CSV (index, angle, std).
  2. Отсечение постоянных участков в начале/конце.
  3. Оценка периода (грубая + уточнение по минимуму дисперсии).
  4. Определение границ ПОЛНЫХ периодов и отбрасывание неполных.
  5. Fold полных периодов на единую фазовую ось.
  6. Глобальный линейный детренд → матрица ошибок.
  7. Робастная статистика (среднее, СКО между периодами, СКО одиночного).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Результат анализа
# ============================================================
@dataclass
class PeriodAnalysisResult:
    period_samples: float
    n_periods: int
    phase_samples: np.ndarray   # фазовая координата внутри периода (в отсчётах)
    phase_deg: np.ndarray       # фазовая координата 0..360°

    mean_error: np.ndarray      # средняя ошибка энкодера, град
    statistical_std: np.ndarray # СКО между периодами, град
    single_frame_std: np.ndarray  # folded СКО одиночного измерения, град
    total_std: np.ndarray       # sqrt(stat² + single²), град

    error_matrix: np.ndarray    # ошибки всех периодов после детренда (n_periods × n_phase)
    outlier_mask: np.ndarray    # True = фазовая точка с выбросами

    fitted_slope: float         # град/отсчёт (знак = направление)
    rotation_direction: str     # "increasing" / "decreasing"
    active_range: Tuple[float, float]  # [x_start, x_end] активного участка


# ============================================================
# Угловые утилиты
# ============================================================
def wrap_deg(a: np.ndarray) -> np.ndarray:
    return np.mod(a, 360.0)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Кратчайшая разность углов a - b, результат в (-180, 180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


def unwrap_deg(a: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(a)))


def circ_mean(a: np.ndarray, axis: int = 0) -> np.ndarray:
    z = np.exp(1j * np.deg2rad(a))
    return wrap_deg(np.rad2deg(np.angle(np.mean(z, axis=axis))))


# ============================================================
# Робастный линейный fit (итеративный, MAD-based)
# ============================================================
def robust_polyfit1(x: np.ndarray, y: np.ndarray,
                    clip: float = 3.0, n_iter: int = 5):
    """Возвращает (slope, intercept, inlier_mask)."""
    mask = np.ones(len(x), dtype=bool)
    slope, intercept = np.polyfit(x, y, 1)
    for _ in range(n_iter):
        r = y - (slope * x + intercept)
        med = np.median(r[mask])
        mad = np.median(np.abs(r[mask] - med))
        s = max(1.4826 * mad, 1e-12)
        mask_new = np.abs(r - med) <= clip * s
        if np.count_nonzero(mask_new) < 3:
            break
        if np.array_equal(mask, mask_new):
            break
        mask = mask_new
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return slope, intercept, mask


# ============================================================
# Чтение CSV
# ============================================================
def read_csv(path: str, sep: str = ","):
    df = pd.read_csv(path, sep=sep, header=None, comment="#", engine="python")
    df = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce").dropna()
    df.columns = ["idx", "angle", "std"]
    df = df.sort_values("idx").reset_index(drop=True)
    x = df["idx"].values.astype(float)
    a = wrap_deg(df["angle"].values.astype(float))
    s = df["std"].values.astype(float)
    return x, a, s


# ============================================================
# Выделение активного участка (отсечение постоянных краёв)
# ============================================================
def trim_constants(x: np.ndarray, angle: np.ndarray, std: np.ndarray,
                   smooth_pct: float = 1.0, threshold_pct: float = 15.0):
    """
    Отсекает начальный и конечный участки, где угол почти не меняется.
    Возвращает обрезанные x, angle, std.
    """
    if len(x) < 5:
        return x, angle, std

    dx = np.diff(x)
    dx = np.where(dx == 0, 1e-12, dx)  # защита от деления на 0
    speed = np.abs(angle_diff(angle[1:], angle[:-1]) / dx)

    # Сглаживание скорости скользящим средним
    w = max(3, int(len(speed) * smooth_pct / 100) | 1)
    speed_s = np.convolve(speed, np.ones(w) / w, mode="same")

    q10, q90 = np.percentile(speed_s, [10, 90])
    thr = q10 + threshold_pct / 100.0 * (q90 - q10)
    active = speed_s >= max(thr, 1e-9)

    # Морфологическое закрытие мелких дыр
    for gap in range(1, w + 1):
        kernel = np.ones(2 * gap + 1) / (2 * gap + 1)
        active = np.convolve(active.astype(float), kernel, mode="same") > 0.5

    idx = np.where(active)[0]
    if len(idx) < 5:
        return x, angle, std

    # speed имеет длину N-1, speed[i] соответствует интервалу [x[i], x[i+1]].
    # Если speed[i] активна, нам нужны точки i и i+1.
    s = idx[0]          # первый активный интервал начинается в точке s
    e = idx[-1] + 1     # последний активный интервал заканчивается в точке e
    e = min(e, len(x) - 1)

    return x[s:e + 1], angle[s:e + 1], std[s:e + 1]


# ============================================================
# Оценка периода
# ============================================================
def estimate_period(x: np.ndarray, angle: np.ndarray,
                    n_hint: float = 10.0,
                    refine_frac: float = 0.05,
                    refine_n: int = 200) -> Tuple[float, float]:
    """
    Грубая оценка периода по наклону unwrapped сигнала + уточнение
    по минимуму дисперсии при фолдинге.

    Возвращает (period, slope_sign).
    """
    u = unwrap_deg(angle)
    slope, _, _ = robust_polyfit1(x, u)
    if abs(slope) < 1e-12:
        raise ValueError("Наклон сигнала ≈ 0, невозможно определить период")

    p0 = 360.0 / abs(slope)
    span = x[-1] - x[0]

    lo = max(p0 * (1 - refine_frac), span / (n_hint * 2))
    hi = min(p0 * (1 + refine_frac), span / max(n_hint / 2, 1))
    if lo >= hi:
        return p0, np.sign(slope)

    dx = np.median(np.diff(x))
    nph = int(np.clip(p0 / dx, 64, 512))

    best_p, best_s = p0, np.inf
    for p in np.linspace(lo, hi, refine_n):
        # Для уточнения — берём только полные периоды внутри данных
        n_full = int((x[-1] - x[0]) / p)
        if n_full < 2:
            continue
        phase = np.linspace(0, p, nph, endpoint=False)
        starts = x[0] + np.arange(n_full) * p
        query = starts[:, None] + phase[None, :]  # (n_full, nph)

        # Все точки query гарантированно в [x[0], x[0] + n_full*p] ⊆ [x[0], x[-1]]
        mat_uw = np.interp(query.ravel(), x, u).reshape(n_full, nph)

        # Детренд каждого периода (вычитаем линейный тренд)
        for k in range(n_full):
            row_x = query[k]
            row_y = mat_uw[k]
            sl = np.polyfit(row_x, row_y, 1)
            mat_uw[k] = row_y - np.polyval(sl, row_x)

        # Дисперсия между периодами в каждой фазовой точке
        sc = np.mean(np.var(mat_uw, axis=0, ddof=1))
        if sc < best_s:
            best_s, best_p = sc, p

    return best_p, np.sign(slope)


# ============================================================
# Нахождение границ полных периодов (через пересечение порога)
# ============================================================
def find_full_periods(x: np.ndarray, angle: np.ndarray,
                      period: float, slope_sign: float) -> np.ndarray:
    """
    Определяет границы полных периодов пилообразного сигнала.

    Полный период = один полный пробег от 0° до 360° (или от 360° до 0°).

    Возвращает массив x-координат начал полных периодов (длина = n_full_periods).
    Каждый полный период: [starts[i], starts[i] + period).
    """
    u = unwrap_deg(angle)
    slope, intercept, _ = robust_polyfit1(x, u)

    # Начало первого полного периода: первый x, где unwrapped фаза
    # пересекает границу 360° после x[0]
    # Фаза внутри периода: phi(x) = (u(x) - u(x0)) mod 360
    # Первый полный период начинается в точке, где phi = 0 (после начального неполного)

    # Проще: используем линейное приближение для определения начал.
    # u(x) ≈ slope * x + intercept
    # Границы периодов: u = intercept + slope*x = N*360 для целых N
    # => x_N = (N*360 - intercept) / slope

    if abs(slope) < 1e-12:
        raise ValueError("Наклон ≈ 0")

    # Номера периодов, покрывающих диапазон данных
    u_start = slope * x[0] + intercept
    u_end = slope * x[-1] + intercept

    if slope > 0:
        n_start = int(np.ceil(u_start / 360.0))
        n_end = int(np.floor(u_end / 360.0))
    else:
        n_start = int(np.floor(u_start / 360.0))
        n_end = int(np.ceil(u_end / 360.0))

    if slope > 0:
        period_boundary_ns = np.arange(n_start, n_end + 1)
    else:
        period_boundary_ns = np.arange(n_end, n_start + 1)

    # x-координаты границ периодов (по линейному приближению)
    boundary_x = (period_boundary_ns * 360.0 - intercept) / slope

    # Оставляем только те границы, которые внутри данных с запасом
    mask = (boundary_x >= x[0]) & (boundary_x <= x[-1])
    boundary_x = boundary_x[mask]

    if len(boundary_x) < 2:
        raise ValueError(f"Недостаточно полных периодов (границ: {len(boundary_x)})")

    # Уточняем границы по реальным данным (ищем ближайшие точки пересечения)
    boundary_x_refined = []
    for bx in boundary_x:
        idx = np.argmin(np.abs(x - bx))
        boundary_x_refined.append(x[idx])

    boundary_x_refined = np.array(boundary_x_refined)

    return boundary_x_refined


# ============================================================
# Fold полных периодов + глобальный детренд + статистика
# ============================================================
def fold_and_detrend(x: np.ndarray, angle: np.ndarray, std: np.ndarray,
                     period: float, slope_sign: float,
                     n_phase: int | None = None,
                     clip: float = 3.0) -> PeriodAnalysisResult:
    """
    Основная функция анализа:
      1. Находит полные периоды (неполные отбрасываются).
      2. Интерполирует каждый полный период на единую фазовую сетку.
      3. Вычитает ГЛОБАЛЬНЫЙ линейный тренд.
      4. Считает робастные статистики по фазовым точкам.
    """
    dx = np.median(np.diff(x))
    if n_phase is None:
        n_phase = max(16, int(round(period / dx)))

    u_global = unwrap_deg(angle)
    slope_val, intercept_val, _ = robust_polyfit1(x, u_global, clip=clip)

    # --- 1. Определяем начала полных периодов ---
    # Полные периоды: начало_i = x[0] + offset + i * period
    # offset выбирается так, чтобы первый полный период целиком лежал внутри данных,
    # и последний тоже.

    # Вместо сложного поиска пересечений используем простой подход:
    # Фаза точки x относительно линейного приближения:
    #   phi(x) = (slope_val * x + intercept_val) mod 360
    # Начало периода: phi = 0
    # По линейному приближению: первое x где phi проходит через 0

    u_start = slope_val * x[0] + intercept_val
    u_end = slope_val * x[-1] + intercept_val

    if slope_val > 0:
        first_boundary_n = int(np.ceil(u_start / 360.0))
        last_boundary_n = int(np.floor(u_end / 360.0))
        boundary_ns = np.arange(first_boundary_n, last_boundary_n + 1)
    else:
        first_boundary_n = int(np.floor(u_start / 360.0))
        last_boundary_n = int(np.ceil(u_end / 360.0))
        boundary_ns = np.arange(first_boundary_n, last_boundary_n - 1, -1)

    boundary_x = (boundary_ns * 360.0 - intercept_val) / slope_val
    # Оставляем только те, что строго внутри данных
    mask = (boundary_x >= x[0]) & (boundary_x <= x[-1])
    boundary_x = np.sort(boundary_x[mask])

    if len(boundary_x) < 2:
        raise ValueError(
            f"Недостаточно полных периодов. "
            f"Найдено границ: {len(boundary_x)}, нужно >= 2."
        )

    # Полные периоды: от boundary_x[i] до boundary_x[i+1]
    n_periods = len(boundary_x) - 1
    period_starts = boundary_x[:-1]
    period_ends = boundary_x[1:]

    # Фактический период — среднее расстояние между границами
    actual_periods = np.diff(boundary_x)
    period_refined = np.median(actual_periods)

    print(f"  Границ периодов найдено: {len(boundary_x)}")
    print(f"  Полных периодов: {n_periods}")
    print(f"  Разброс длин периодов: "
          f"{actual_periods.min():.2f} .. {actual_periods.max():.2f} "
          f"(медиана {period_refined:.2f})")

    # --- 2. Фазовая сетка ---
    phase_frac = np.linspace(0, 1, n_phase, endpoint=False)  # 0..1

    # --- 3. Интерполяция каждого полного периода ---
    angle_mat = np.empty((n_periods, n_phase))
    std_mat = np.empty((n_periods, n_phase))
    query_x_mat = np.empty((n_periods, n_phase))

    for k in range(n_periods):
        xs = period_starts[k]
        xe = period_ends[k]
        local_period = xe - xs

        # Абсолютные x-координаты для фазовых точек этого периода
        query_x = xs + phase_frac * local_period
        query_x_mat[k] = query_x

        # Интерполяция unwrapped угла и std
        angle_mat[k] = np.interp(query_x, x, u_global)
        std_mat[k] = np.interp(query_x, x, std)

    # --- 4. Глобальный линейный детренд ---
    # Вычитаем из каждой точки её ожидаемое значение по глобальному fit
    trend = slope_val * query_x_mat + intercept_val
    error_mat = angle_mat - trend  # (n_periods, n_phase) — ошибки в градусах

    # --- 5. Робастная статистика по каждой фазовой точке ---
    mean_err = np.empty(n_phase)
    stat_std = np.empty(n_phase)
    outlier = np.zeros(n_phase, dtype=bool)

    for j in range(n_phase):
        col = error_mat[:, j]
        col_mask = np.ones(n_periods, dtype=bool)

        # Итеративное отсечение выбросов
        for _ in range(5):
            vals = col[col_mask]
            if len(vals) < 2:
                break
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            s = max(1.4826 * mad, 1e-12)
            col_mask_new = np.abs(col - med) <= clip * s
            if np.count_nonzero(col_mask_new) < 2:
                break
            if np.array_equal(col_mask, col_mask_new):
                break
            col_mask = col_mask_new

        n_inliers = np.count_nonzero(col_mask)
        mean_err[j] = np.mean(col[col_mask])
        stat_std[j] = np.std(col[col_mask], ddof=1) if n_inliers > 1 else 0.0
        if np.count_nonzero(~col_mask) > 0:
            outlier[j] = True

    # --- 6. СКО одиночного измерения (среднеквадратичное по периодам) ---
    single_std = np.sqrt(np.mean(std_mat ** 2, axis=0))
    total_std = np.sqrt(stat_std ** 2 + single_std ** 2)

    # --- 7. Фазовая ось в градусах ---
    phase_samples = phase_frac * period_refined
    if slope_sign >= 0:
        phase_deg = phase_frac * 360.0
    else:
        phase_deg = (1.0 - phase_frac) * 360.0
        # Переупорядочиваем по возрастанию phase_deg
        sort_idx = np.argsort(phase_deg)
        phase_deg = phase_deg[sort_idx]
        phase_samples = phase_samples[sort_idx]
        mean_err = mean_err[sort_idx]
        stat_std = stat_std[sort_idx]
        single_std = single_std[sort_idx]
        total_std = total_std[sort_idx]
        outlier = outlier[sort_idx]
        error_mat = error_mat[:, sort_idx]

    return PeriodAnalysisResult(
        period_samples=period_refined,
        n_periods=n_periods,
        phase_samples=phase_samples,
        phase_deg=phase_deg,
        mean_error=mean_err,
        statistical_std=stat_std,
        single_frame_std=single_std,
        total_std=total_std,
        error_matrix=error_mat,
        outlier_mask=outlier,
        fitted_slope=slope_val,
        rotation_direction="increasing" if slope_sign >= 0 else "decreasing",
        active_range=(float(x[0]), float(x[-1])),
    )


# ============================================================
# Полный анализ
# ============================================================
def analyze(csv_path: str, sep: str = ",",
            n_hint: float = 10.0,
            min_p: float | None = None,
            max_p: float | None = None,
            n_phase: int | None = None,
            clip: float = 3.0) -> PeriodAnalysisResult:

    x, angle, std = read_csv(csv_path, sep=sep)
    print(f"Загружено точек: {len(x)}")

    x, angle, std = trim_constants(x, angle, std)
    print(f"После trim_constants: {len(x)} точек, "
          f"x ∈ [{x[0]:.1f}, {x[-1]:.1f}]")

    period, slope_sign = estimate_period(x, angle, n_hint=n_hint)
    print(f"Оценка периода: {period:.4f} отсч., "
          f"направление: {'↑' if slope_sign > 0 else '↓'}")

    if min_p and period < min_p:
        print(f"  Период {period:.2f} < min_p={min_p}, корректируем")
        period = min_p
    if max_p and period > max_p:
        print(f"  Период {period:.2f} > max_p={max_p}, корректируем")
        period = max_p

    return fold_and_detrend(x, angle, std, period, slope_sign,
                            n_phase=n_phase, clip=clip)


# ============================================================
# Визуализация
# ============================================================
def plot_result(r: PeriodAnalysisResult, out_path: str | None = None):
    ph = r.phase_deg
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   sharex=True, constrained_layout=True)

    # --- Верхний график: средняя ошибка ± СКО ---
    ax1.fill_between(ph,
                     r.mean_error - r.total_std,
                     r.mean_error + r.total_std,
                     color="tab:orange", alpha=0.2, label="± суммарное СКО")
    ax1.fill_between(ph,
                     r.mean_error - r.single_frame_std,
                     r.mean_error + r.single_frame_std,
                     color="tab:blue", alpha=0.3, label="± СКО одиночного")
    ax1.plot(ph, r.mean_error, "k", lw=2, label="Средняя ошибка")
    ax1.axhline(0, color="gray", lw=0.5, ls="--")
    ax1.set_ylabel("Ошибка энкодера, град")
    ax1.set_title(
        f"Период = {r.period_samples:.4f} отсч. | "
        f"N = {r.n_periods} полных периодов | "
        f"{r.rotation_direction}"
    )
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(-0.5, 0.5)
    ax1.grid(True, alpha=0.3)

    # --- Нижний график: СКО ---
    ax2.plot(ph, r.single_frame_std, color="tab:blue",
             label="СКО одиночного")
    ax2.plot(ph, r.statistical_std, color="tab:green",
             label="Стат. СКО между периодами")
    ax2.plot(ph, r.total_std, color="tab:orange",
             label="Суммарное СКО")
    ax2.set_xlabel("Угол внутри периода, град")
    ax2.set_ylabel("СКО, град")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0, 0.5)
    ax2.set_xticks(np.arange(0, 361, 30))
    ax2.grid(True, alpha=0.3)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# Сохранение CSV
# ============================================================
def save_csv(r: PeriodAnalysisResult, path: str):
    pd.DataFrame({
        "sample_in_period": r.phase_samples,
        "angle_deg": r.phase_deg,
        "mean_error_deg": r.mean_error,
        "stat_std_deg": r.statistical_std,
        "single_std_deg": r.single_frame_std,
        "total_std_deg": r.total_std,
        "outlier": r.outlier_mask.astype(int),
    }).to_csv(path, index=False)


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description="Анализ периодического сигнала абсолютного энкодера"
    )
    p.add_argument("csv", help="Входной CSV: index, angle, std")
    p.add_argument("--sep", default=",", help="Разделитель CSV")
    p.add_argument("--n-hint", type=float, default=10.0,
                   help="Ожидаемое кол-во периодов (подсказка)")
    p.add_argument("--min-period", type=float, default=None)
    p.add_argument("--max-period", type=float, default=None)
    p.add_argument("--n-phase", type=int, default=None,
                   help="Кол-во фазовых точек (авто если не задано)")
    p.add_argument("--clip", type=float, default=3.0,
                   help="Порог отсечения выбросов (в σ)")
    p.add_argument("--plot", default="encoder_profile.png")
    p.add_argument("--out-csv", default="encoder_profile.csv")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    r = analyze(a.csv, sep=a.sep, n_hint=a.n_hint,
                min_p=a.min_period, max_p=a.max_period,
                n_phase=a.n_phase, clip=a.clip)

    print(f"\n=== Результат ===")
    print(f"Период:       {r.period_samples:.4f} отсч.")
    print(f"Направление:  {r.rotation_direction}")
    print(f"Периодов:     {r.n_periods}")
    print(f"Фазовых точек: {len(r.phase_deg)}")
    print(f"Выбросов:     {np.count_nonzero(r.outlier_mask)}/{len(r.outlier_mask)}")
    print(f"Макс |ошибка|: {np.max(np.abs(r.mean_error)):.6f}°")
    print(f"RMS ошибки:   {np.sqrt(np.mean(r.mean_error**2)):.6f}°")
    print(f"Медиана стат. СКО: {np.median(r.statistical_std):.6f}°")

    save_csv(r, a.out_csv)
    plot_result(r, out_path=a.plot)
    print(f"\nСохранено: {a.out_csv}, {a.plot}")

    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
