from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class PeriodAnalysisResult:
    period_samples: float
    n_periods: int
    phase_samples: np.ndarray

    mean_error: np.ndarray          # средняя ошибка энкодера, град
    statistical_std: np.ndarray     # СКО между периодами, град
    single_frame_std: np.ndarray    # folded СКО одиночного измерения, град
    total_std: np.ndarray           # sqrt(stat^2 + single^2), град

    error_matrix: np.ndarray        # ошибки всех периодов после детренда
    outlier_mask: np.ndarray        # True = фазовая точка с выбросами

    fitted_slope: float             # град/отсчет
    rotation_direction: str         # "increasing" / "decreasing"
    active_range: Tuple[float, float]


# ============================================================
# Угловые утилиты
# ============================================================
def wrap_deg(a):
    return np.mod(a, 360.0)

def angle_diff(a, b):
    return ((a - b + 180.0) % 360.0) - 180.0

def unwrap_deg(a):
    return np.rad2deg(np.unwrap(np.deg2rad(a)))

def circ_mean(a, axis=0):
    z = np.exp(1j * np.deg2rad(a))
    return wrap_deg(np.rad2deg(np.angle(np.mean(z, axis=axis))))

def circ_interp(x, angle, xq):
    z = np.exp(1j * np.deg2rad(angle))
    return wrap_deg(np.rad2deg(np.angle(
        np.interp(xq, x, z.real) + 1j * np.interp(xq, x, z.imag)
    )))


# ============================================================
# Робастный линейный fit (итеративный, MAD-based)
# ============================================================
def robust_polyfit1(x, y, clip=3.0, n_iter=5):
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
def read_csv(path, sep=","):
    df = pd.read_csv(path, sep=sep, header=None, comment="#", engine="python")
    df = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce").dropna()
    df.columns = ["idx", "angle", "std"]
    df = df.sort_values("idx").reset_index(drop=True)
    x = df["idx"].values.astype(float)
    a = wrap_deg(df["angle"].values.astype(float))
    s = df["std"].values.astype(float)
    return x, a, s


# ============================================================
# Выделение активного участка (отсечение констант)
# ============================================================
def trim_constants(x, angle, std, smooth_pct=1.0, threshold_pct=15.0):
    dx = np.diff(x)
    speed = np.abs(angle_diff(angle[1:], angle[:-1]) / dx)
    w = max(3, int(len(speed) * smooth_pct / 100) | 1)
    speed_s = np.convolve(speed, np.ones(w)/w, mode="same")
    q10, q90 = np.percentile(speed_s, [10, 90])
    thr = q10 + threshold_pct / 100 * (q90 - q10)
    active = speed_s >= max(thr, 1e-9)
    for gap in range(1, w + 1):
        active = np.convolve(active.astype(float), np.ones(2*gap+1)/(2*gap+1), mode="same") > 0.5
    idx = np.where(active)[0]
    if len(idx) < 5:
        return x, angle, std
    s, e = idx[0], idx[-1] + 1
    return x[s:e+1], angle[s:e+1], std[s:e+1]


# ============================================================
# Оценка периода
# ============================================================
def estimate_period(x, angle, n_hint=10.0, refine_frac=0.05, refine_n=200):
    u = unwrap_deg(angle)
    slope, _, _ = robust_polyfit1(x, u)
    if abs(slope) < 1e-12:
        raise ValueError("slope ~ 0")
    p0 = 360.0 / abs(slope)
    span = x[-1] - x[0]
    lo = max(p0 * (1 - refine_frac), span / (n_hint * 2))
    hi = min(p0 * (1 + refine_frac), span / max(n_hint / 2, 1))
    if lo >= hi:
        return p0, slope

    dx = np.median(np.diff(x))
    nph = int(np.clip(p0 / dx, 64, 512))
    best_p, best_s = p0, np.inf
    for p in np.linspace(lo, hi, refine_n):
        ph, q = _grid(x, p, nph)
        mat = circ_interp(x, angle, q.ravel()).reshape(q.shape)
        if mat.shape[0] < 2:
            continue
        m = circ_mean(mat, axis=0)
        res = angle_diff(mat, m[None, :])
        sc = np.mean(np.var(res, axis=0, ddof=1))
        if sc < best_s:
            best_s, best_p = sc, p
    return best_p, slope


def _grid(x, period, n_phase, start=None):
    dx = np.median(np.diff(x))
    if start is None:
        start = x[0]
    phase = np.arange(n_phase) * (period / n_phase)
    n_per = int((x[-1] - start - phase[-1]) / period) + 1
    if n_per < 1:
        raise ValueError("Недостаточно данных для одного периода")
    q = start + np.arange(n_per)[:, None] * period + phase[None, :]
    return phase, q


# ============================================================
# Fold + детрендирование по глобальным координатам + ре-фазировка
# ============================================================
def fold_and_detrend(x, angle, std, period, slope_sign, n_phase=None, clip=3.0):
    dx = np.median(np.diff(x))
    if n_phase is None:
        n_phase = max(16, int(round(period / dx)))

    phase, query = _grid(x, period, n_phase)
    n_per = query.shape[0]

    # --- FIX 1: интерполируем unwrapped сигнал напрямую, без повторного wrap ---
    # Это избегает артефактов двойного wrap/unwrap через circ_interp
    u_global = unwrap_deg(angle)
    angle_mat_uw = np.interp(query.ravel(), x, u_global).reshape(n_per, n_phase)

    # СКО одиночного измерения интерполируем линейно
    std_mat = np.interp(query.ravel(), x, std).reshape(n_per, n_phase)

    # --- FIX 2: детрендируем по глобальным координатам (query[k]), не по phase ---
    error_mat = np.empty_like(angle_mat_uw)
    for k in range(n_per):
        row = angle_mat_uw[k]           # уже unwrapped, без дополнительного unwrap
        global_coords = query[k]        # реальные координаты: start + k*period + phase
        sl, ic, _ = robust_polyfit1(global_coords, row, clip=clip)
        error_mat[k] = row - (sl * global_coords + ic)

    # робастное среднее и СКО по столбцам
    mean_err = np.empty(n_phase)
    stat_std = np.empty(n_phase)
    outlier = np.zeros(n_phase, dtype=bool)

    for j in range(n_phase):
        col = error_mat[:, j]
        mask = np.ones(n_per, dtype=bool)
        for _ in range(5):
            vals = col[mask]
            if len(vals) < 2:
                break
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            s = max(1.4826 * mad, 1e-12)
            mask_new = np.abs(col - med) <= clip * s
            if np.count_nonzero(mask_new) < 2:
                break
            if np.array_equal(mask, mask_new):
                break
            mask = mask_new
        mean_err[j] = np.mean(col[mask])
        stat_std[j] = np.std(col[mask], ddof=1) if np.count_nonzero(mask) > 1 else 0.0
        if np.count_nonzero(~mask) > 0:
            outlier[j] = True

    single_std = np.sqrt(np.mean(std_mat ** 2, axis=0))
    total_std = np.sqrt(stat_std ** 2 + single_std ** 2)

    # --- FIX 3: ре-фазировка — сдвигаем и данные, и phase_samples ---
    # Определяем скачок wrap на границе по wrapped среднему
    angle_mat_wrapped = wrap_deg(angle_mat_uw)
    mean_raw = circ_mean(angle_mat_wrapped, axis=0)
    diffs = np.diff(mean_raw)
    if slope_sign >= 0:
        idx = int(np.argmin(diffs))
        do_shift = diffs[idx] < -180
    else:
        idx = int(np.argmax(diffs))
        do_shift = diffs[idx] > 180

    if do_shift:
        sh = idx + 1
        # Сдвигаем phase_samples циклически, сохраняя монотонность
        # Новая нулевая точка — phase[sh], диапазон переносится в [0, period)
        phase_offset = phase[sh]
        phase = (phase - phase_offset) % period
        # Сортируем по новой фазе — восстанавливаем монотонность
        sort_idx = np.argsort(phase)
        phase = phase[sort_idx]
        for arr in [mean_err, stat_std, single_std, total_std, outlier]:
            arr[:] = arr[sort_idx]
        error_mat = error_mat[:, sort_idx]

    return PeriodAnalysisResult(
        period_samples=period,
        n_periods=n_per,
        phase_samples=phase,
        mean_error=mean_err,
        statistical_std=stat_std,
        single_frame_std=single_std,
        total_std=total_std,
        error_matrix=error_mat,
        outlier_mask=outlier,
        fitted_slope=slope_sign,
        rotation_direction="increasing" if slope_sign >= 0 else "decreasing",
        active_range=(float(x[0]), float(x[-1])),
    )


# ============================================================
# Полный анализ
# ============================================================
def analyze(csv_path, sep=",", n_hint=10.0, min_p=None, max_p=None,
            n_phase=None, clip=3.0):
    x, angle, std = read_csv(csv_path, sep=sep)
    x, angle, std = trim_constants(x, angle, std)
    period, slope = estimate_period(x, angle, n_hint=n_hint)
    if min_p and period < min_p:
        period = min_p
    if max_p and period > max_p:
        period = max_p
    return fold_and_detrend(x, angle, std, period, slope, n_phase=n_phase, clip=clip)


# ============================================================
# Визуализация: 2 графика
# ============================================================
def plot_result(r: PeriodAnalysisResult, out_path=None):
    x = r.phase_samples
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    sharex=True, constrained_layout=True)

    ax1.fill_between(x, r.mean_error - r.total_std, r.mean_error + r.total_std,
                     color="tab:orange", alpha=0.2, label="± суммарное СКО")
    ax1.fill_between(x, r.mean_error - r.single_frame_std,
                     r.mean_error + r.single_frame_std,
                     color="tab:blue", alpha=0.3, label="± СКО одиночного")
    ax1.plot(x, r.mean_error, "k", lw=2, label="Средняя ошибка")

    ax1.axhline(0, color="gray", lw=0.5, ls="--")
    ax1.set_ylabel("Ошибка энкодера, град")
    ax1.set_title(f"Период={r.period_samples:.4f} отсч. | "
                  f"N={r.n_periods} | {r.rotation_direction}")
    ax1.legend(fontsize=8)
    ax1.set_ylim([-0.25, 0.25])
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, r.single_frame_std, color="tab:blue", label="СКО одиночного")
    ax2.plot(x, r.statistical_std, color="tab:green", label="Стат. СКО между периодами")
    ax2.plot(x, r.total_std, color="tab:orange", label="Суммарное СКО")
    ax2.set_xlabel("Отсчёт внутри периода")
    ax2.set_ylabel("СКО, град")
    ax2.legend(fontsize=8)
    ax2.set_ylim([0, 0.075])
    ax2.grid(True, alpha=0.3)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# Сохранение CSV
# ============================================================
def save_csv(r: PeriodAnalysisResult, path):
    pd.DataFrame({
        "sample_in_period": r.phase_samples,
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
    p = argparse.ArgumentParser(description="Анализ абсолютного энкодера")
    p.add_argument("csv")
    p.add_argument("--sep", default=",")
    p.add_argument("--n-hint", type=float, default=10.0)
    p.add_argument("--min-period", type=float, default=None)
    p.add_argument("--max-period", type=float, default=None)
    p.add_argument("--n-phase", type=int, default=None)
    p.add_argument("--clip", type=float, default=3.0)
    p.add_argument("--plot", default="encoder_profile.png")
    p.add_argument("--out-csv", default="encoder_profile.csv")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    r = analyze(a.csv, sep=a.sep, n_hint=a.n_hint,
                min_p=a.min_period, max_p=a.max_period,
                n_phase=a.n_phase, clip=a.clip)

    print(f"Период:      {r.period_samples:.4f} отсч.")
    print(f"Направление: {r.rotation_direction}")
    print(f"Периодов:    {r.n_periods}")
    print(f"Выбросов:    {np.count_nonzero(r.outlier_mask)}/{len(r.outlier_mask)}")

    save_csv(r, a.out_csv)
    plot_result(r, out_path=a.plot)
    print(f"Сохранено: {a.out_csv}, {a.plot}")

    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
