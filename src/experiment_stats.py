from itertools import product
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.blais_rioux import detect_edges_and_recover_bits, BRConfig
from src.ccd_simulator import simulate_ccd, SimulatorConfig


# ============================================================
# 1. Построение дискретной сетки параметров
# ============================================================

def expand_grid(param_grid: dict) -> list[dict]:
    """
    Преобразует словарь вида
        {
            "noise_sigma_adu": [0.0, 1.0, 2.0],
            "sigma_blur_px": [0.5, 1.0],
        }
    в список конфигураций:
        [
            {"noise_sigma_adu": 0.0, "sigma_blur_px": 0.5},
            {"noise_sigma_adu": 0.0, "sigma_blur_px": 1.0},
            {"noise_sigma_adu": 1.0, "sigma_blur_px": 0.5},
            ...
        ]
    """
    keys = list(param_grid.keys())
    # Конвертируем numpy массивы (и вообще любые итерируемые) в list
    values = [list(v) if hasattr(v, '__iter__') else [v] for v in
              param_grid.values()]

    configs = []
    for combo in product(*values):
        configs.append(dict(zip(keys, combo)))

    return configs

# ============================================================
# 2. Заглушка: сюда вы подставите свою реализацию
# ============================================================

def run_single_test(config: dict, repeat_id: int):
    n_bits = 30
    delta = 5
    vignette_strength = 0.25
    distort_coeff = 0.1
    oversample = 32
    adc_bits = 12

    sim_config = SimulatorConfig(
        seed=repeat_id,
        n_bits=n_bits,
        vignette_strength=vignette_strength,
        distort_coeff=distort_coeff,
        oversample=oversample,
        adc_bits=adc_bits,
        n_pixels=round((n_bits + delta) * config["bit_width_px"]),
        noise_sigma_adu=config["noise_sigma_adu"],
        sigma_blur_px=config["sigma_blur_ratio"] * config["bit_width_px"],
        bit_width_px=config["bit_width_px"],
    )

    peak_threshold_rel = 0.25
    min_edge_distance_factor = 0.5
    smoothing_sigma = 0.2
    minmax_window_px = 50

    br_config = BRConfig(
        peak_threshold_rel=peak_threshold_rel,
        min_edge_distance_factor=min_edge_distance_factor,
        smoothing_sigma=smoothing_sigma,
        minmax_window_px=minmax_window_px,
        filter_order=config["filter_order"],
        bit_width_px=config["bit_width_px"],
    )

    sim_result = simulate_ccd(sim_config)

    detection_result = detect_edges_and_recover_bits(
        sim_result.adc_signal,
        sim_result.bits,
        sim_result.true_edges,
        br_config,
    )

    return {
        "sigma": float(detection_result.rms_edge_error),   # px
        "accuracy": float(detection_result.accuracy),
    }


# ============================================================
# 3. Накопление сырых результатов
# ============================================================

def collect_raw_results(
    param_grid: dict,
    n_repeats: int,
    run_single_test_fn: Callable,
    n_trials: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Для каждой конфигурации:
      - для каждого repeat_id выполняется серия из n_trials прогонов
        с одним и тем же repeat_id;
      - результаты серии усредняются;
      - одна строка в raw_df = один repeat_id, усредненный по n_trials.
    """
    configs = expand_grid(param_grid)
    rows = []

    total_configs = len(configs)

    for config_id, config in enumerate(configs):
        if verbose:
            print(f"[config {config_id + 1}/{total_configs}] {config}")

        for repeat_id in range(n_repeats):
            sigma_values = []
            accuracy_values = []
            error_messages = []

            for trial_id in range(n_trials):
                try:
                    result = run_single_test_fn(config, repeat_id)

                    sigma_values.append(float(result["sigma"]))
                    accuracy_values.append(float(result["accuracy"]))

                except Exception as e:
                    error_messages.append(f"trial {trial_id}: {type(e).__name__}: {e}")

            row = {
                "config_id": config_id,
                "repeat_id": repeat_id,
                **config,
                "n_trials": n_trials,
                "n_trials_ok": len(sigma_values),
                "n_trials_failed": n_trials - len(sigma_values),
            }

            if len(sigma_values) > 0:
                row["sigma"] = float(np.mean(sigma_values))
                row["accuracy"] = float(np.mean(accuracy_values))
                row["status"] = "ok"
                row["error_text"] = ""
            else:
                row["sigma"] = np.nan
                row["accuracy"] = np.nan
                row["status"] = "failed"
                row["error_text"] = " | ".join(error_messages[:5])

            rows.append(row)

    return pd.DataFrame(rows)


def to_records_array(df: pd.DataFrame) -> np.ndarray:
    """
    Если нужен именно numpy-массив записей.
    """
    return df.to_records(index=False)


# ============================================================
# 4. Вычисление статистики по конфигурациям
# ============================================================

def compute_statistics(
    raw_df: pd.DataFrame,
    param_names: list[str],
) -> pd.DataFrame:
    """
    Статистика по конфигурациям.

    Одна строка raw_df = один repeat_id, уже усредненный по n_trials.
    Здесь считаем статистику по repeat_id внутри одной конфигурации.
    """
    group_cols = ["config_id"] + param_names

    # Сколько repeat'ов было всего на конфигурацию
    repeats_df = (
        raw_df
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_repeats")
    )

    # Только успешные repeat'ы
    ok_df = raw_df[raw_df["status"] == "ok"].copy()

    stats_df = (
        ok_df
        .groupby(group_cols, dropna=False)
        .agg(
            n_ok_repeats=("status", "count"),

            sigma_mean=("sigma", "mean"),
            sigma_std=("sigma", "std"),
            sigma_min=("sigma", "min"),
            sigma_max=("sigma", "max"),

            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),

            n_trials_ok_total=("n_trials_ok", "sum"),
            n_trials_failed_total=("n_trials_failed", "sum"),
        )
        .reset_index()
    )

    stats_df = repeats_df.merge(stats_df, on=group_cols, how="left")

    stats_df["n_ok_repeats"] = stats_df["n_ok_repeats"].fillna(0).astype(int)
    stats_df["n_failed_repeats"] = stats_df["n_repeats"] - stats_df["n_ok_repeats"]

    stats_df["n_trials_ok_total"] = stats_df["n_trials_ok_total"].fillna(0).astype(int)
    stats_df["n_trials_failed_total"] = stats_df["n_trials_failed_total"].fillna(0).astype(int)

    # std при одном значении будет NaN -> заменяем на 0
    stats_df["sigma_std"] = stats_df["sigma_std"].fillna(0.0)
    stats_df["accuracy_std"] = stats_df["accuracy_std"].fillna(0.0)

    return stats_df.sort_values("config_id").reset_index(drop=True)


# ============================================================
# 5. Сохранение таблиц
# ============================================================

def save_results(
    raw_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: str,
) -> None:
    """
    Сохраняет сырые данные и статистику в CSV.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(out_path / "raw_results.csv", index=False)
    stats_df.to_csv(out_path / "stats_results.csv", index=False)


# ============================================================
# 6. Фильтрация для построения проекций
# ============================================================

def filter_by_params(
    df: pd.DataFrame,
    fixed_params: Optional[dict] = None,
    param_grid: Optional[dict] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Фильтрует DataFrame по fixed_params.
    Если задан param_grid, значения fixed_params сначала
    привязываются к ближайшим существующим значениям сетки.
    """
    if fixed_params is None:
        return df.copy(), fixed_params

    snapped_params = snap_fixed_params_to_grid(fixed_params, param_grid)

    if verbose and snapped_params != fixed_params:
        print("fixed_params:")
        print("  requested:", fixed_params)
        print("  snapped:  ", snapped_params)

    out = df.copy()
    for key, value in snapped_params.items():
        out = out[out[key] == value]

    return out, snapped_params


def snap_value_to_grid(value, grid_values):
    """
    Возвращает ближайшее существующее значение из сетки.

    Для чисел - ближайшее по модулю.
    Для нечисловых значений - точное совпадение, если есть,
    иначе первое значение из списка.
    """
    arr = np.asarray(list(grid_values))

    # Числовой случай
    if np.issubdtype(arr.dtype, np.number) and isinstance(value, (int, float, np.integer, np.floating)):
        idx = np.argmin(np.abs(arr.astype(float) - float(value)))
        return arr[idx].item()

    # Нечисловой случай
    for v in arr:
        if v == value:
            return v.item() if hasattr(v, "item") else v

    v = arr[0]
    return v.item() if hasattr(v, "item") else v


def snap_fixed_params_to_grid(fixed_params: Optional[dict], param_grid: Optional[dict]) -> dict:
    """
    Для каждого параметра из fixed_params выбирает ближайшее значение из param_grid.
    Если параметра нет в param_grid, оставляет как есть.
    """
    if fixed_params is None:
        return {}

    if param_grid is None:
        return dict(fixed_params)

    snapped = {}
    for key, value in fixed_params.items():
        if key in param_grid:
            snapped[key] = snap_value_to_grid(value, param_grid[key])
        else:
            snapped[key] = value

    return snapped

# ============================================================
# 7. График 1D: качество(x)
# ============================================================

def plot_sigma_1d(
    stats_df: pd.DataFrame,
    x_param: str,
    fixed_params: Optional[dict] = None,
    curve_param: Optional[str] = None,
    sigma_col: str = "sigma_mean",
    sigma_std_col: str = "sigma_std",
    param_grid: Optional[dict] = None,
    ax=None,
):
    data, snapped_params = filter_by_params(stats_df, fixed_params, param_grid=param_grid, verbose=True)

    if data.empty:
        raise ValueError("Нет данных после фильтрации")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    def _plot_group(group: pd.DataFrame, label_suffix: str):
        group = group.sort_values(x_param)

        x = group[x_param].to_numpy(dtype=float)
        y_sigma = group[sigma_col].to_numpy(dtype=float)
        s_sigma = group[sigma_std_col].to_numpy(dtype=float)

        (line,) = ax.plot(
            x, y_sigma,
            marker="o",
            label=f"sigma, {label_suffix}" if label_suffix else "sigma",
        )
        color = line.get_color()

        ax.fill_between(
            x,
            y_sigma - s_sigma,
            y_sigma + s_sigma,
            alpha=0.18,
            color=color,
        )

    if curve_param is None:
        if data[x_param].duplicated().any():
            raise ValueError(
                f"После фильтрации найдено несколько точек с одинаковым {x_param}. "
                f"Зафиксируйте больше параметров или используйте curve_param."
            )
        _plot_group(data, "")
    else:
        for curve_value, group in data.groupby(curve_param, dropna=False):
            if group[x_param].duplicated().any():
                raise ValueError(
                    f"Для {curve_param}={curve_value} найдено несколько точек с одинаковым {x_param}. "
                    f"Зафиксируйте больше параметров."
                )
            _plot_group(group, f"{curve_param}={curve_value}")

    ax.set_xlabel(x_param)
    ax.set_ylabel("sigma [px]")
    subtitle = ", ".join(f"{k}={v}" for k, v in snapped_params.items())
    ax.set_title(f"sigma vs {x_param}\n({subtitle})")
    ax.grid(True, alpha=0.3)

    ax.legend(loc="best")

    return ax


# ============================================================
# 8. 2D heatmap
# ============================================================

def plot_quality_heatmap(
    stats_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    fixed_params: Optional[dict] = None,
    value_col: str = "sigma_mean",
    param_grid: Optional[dict] = None,
    ax=None,
    cmap: str = "viridis",
):
    data, snapped_params = filter_by_params(stats_df, fixed_params, param_grid=param_grid, verbose=True)

    if data.empty:
        raise ValueError("Нет данных после фильтрации")

    if data.duplicated(subset=[x_param, y_param]).any():
        raise ValueError(
            f"После фильтрации есть несколько строк на одну пару ({x_param}, {y_param}). "
            f"Зафиксируйте больше параметров."
        )

    pivot = data.pivot(index=y_param, columns=x_param, values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    subtitle = ", ".join(f"{k}={v}" for k, v in snapped_params.items())
    ax.set_title(f"{value_col}({x_param}, {y_param})\n({subtitle})")

    plt.colorbar(im, ax=ax)

    return ax


# ============================================================
# 9. 3D surface
# ============================================================

def plot_quality_surface(
    stats_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    fixed_params: Optional[dict] = None,
    value_col: str = "sigma_mean",
    param_grid: Optional[dict] = None,
    cmap: str = "viridis",
    ax=None,
):
    data = filter_by_params(stats_df, fixed_params, param_grid=param_grid, verbose=True)

    if data.empty:
        raise ValueError("Нет данных после фильтрации")

    if data.duplicated(subset=[x_param, y_param]).any():
        raise ValueError(
            f"После фильтрации есть несколько строк на одну пару ({x_param}, {y_param}). "
            f"Зафиксируйте больше параметров."
        )

    pivot = data.pivot(index=y_param, columns=x_param, values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)

    x_vals = pivot.columns.to_numpy(dtype=float)
    y_vals = pivot.index.to_numpy(dtype=float)
    z_vals = pivot.to_numpy(dtype=float)

    X, Y = np.meshgrid(x_vals, y_vals)

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, z_vals, cmap=cmap, edgecolor="none")

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(value_col)
    ax.set_title(f"{value_col}({x_param}, {y_param})")

    ax.figure.colorbar(surf, ax=ax, shrink=0.7, pad=0.1)

    return ax

# ============================================================
# 10. Пример использования
# ============================================================

if __name__ == "__main__":
    # Сетка параметров.
    # Сюда можно включать любые параметры симуляции и/или обработки,
    # которые вы хотите варьировать.
    param_grid = {
        "bit_width_px": np.round(np.arange(5, 15.5, 0.5),decimals=1),
        "sigma_blur_ratio": np.round(np.arange(0.05, 0.21, 0.01), decimals=2),
        "noise_sigma_adu": np.arange(0, 300, 50),
        "filter_order": [2, 4]
    }

    compute = False

    if compute:
        n_repeats = 10

        # 1) сбор сырых результатов
        raw_df = collect_raw_results(
            param_grid=param_grid,
            n_repeats=10,
            n_trials=3,
            run_single_test_fn=run_single_test,
            verbose=True,
        )
        # 2) расчет статистики
        stats_df = compute_statistics(
            raw_df=raw_df,
            param_names=["bit_width_px", "sigma_blur_ratio", "noise_sigma_adu",
                         "filter_order"],
        )

        # 3) сохранение
        save_results(raw_df, stats_df, out_dir="results")

        # Если нужен именно массив записей:
        raw_array = to_records_array(raw_df)
        stats_array = to_records_array(stats_df)

        print(raw_df.head())
        print(stats_df.head())

    else:
        raw_df = pd.read_csv("./results/raw_results.csv")
        stats_df = pd.read_csv("./results/stats_results.csv")

    plot_sigma_1d(
        stats_df=stats_df,
        x_param="noise_sigma_adu",
        curve_param="filter_order",
        fixed_params={
            "bit_width_px": 6,
            "sigma_blur_ratio": 0.05,
        },
        param_grid=param_grid,
    )

    plot_sigma_1d(
        stats_df=stats_df,
        x_param="bit_width_px",
        curve_param="filter_order",
        fixed_params={
            "noise_sigma_adu": 100,
            "sigma_blur_ratio": 0.2,
        },
        param_grid=param_grid,
    )

    plot_sigma_1d(
        stats_df=stats_df,
        x_param="bit_width_px",
        curve_param="filter_order",
        fixed_params={
            "noise_sigma_adu": 150,
            "sigma_blur_ratio": 0.2,
        },
        param_grid=param_grid,
    )
    plot_sigma_1d(
        stats_df=stats_df,
        x_param="bit_width_px",
        curve_param="filter_order",
        fixed_params={
            "noise_sigma_adu": 50,
            "sigma_blur_ratio": 0.2,
        },
        param_grid=param_grid,
    )

    plot_sigma_1d(
        stats_df=stats_df,
        x_param="bit_width_px",
        curve_param="filter_order",
        fixed_params={
            "noise_sigma_adu": 0,
            "sigma_blur_ratio": 0.2,
        },
        param_grid=param_grid,
    )

    plot_sigma_1d(
        stats_df=stats_df,
        x_param="bit_width_px",
        curve_param="filter_order",
        fixed_params={
            "noise_sigma_adu": 100,
            "sigma_blur_ratio": 0.05,
        },
        param_grid=param_grid,
    )

    plot_sigma_1d(
        stats_df=stats_df,
        x_param="bit_width_px",
        curve_param="filter_order",
        fixed_params={
            "noise_sigma_adu": 150,
            "sigma_blur_ratio": 0.05,
        },
        param_grid=param_grid,
    )
    plot_sigma_1d(
        stats_df=stats_df,
        x_param="bit_width_px",
        curve_param="filter_order",
        fixed_params={
            "noise_sigma_adu": 50,
            "sigma_blur_ratio": 0.05,
        },
        param_grid=param_grid,
    )

    plot_sigma_1d(
        stats_df=stats_df,
        x_param="bit_width_px",
        curve_param="filter_order",
        fixed_params={
            "noise_sigma_adu": 0,
            "sigma_blur_ratio": 0.05,
        },
        param_grid=param_grid,
    )

    plot_sigma_1d(
        stats_df=stats_df,
        x_param="sigma_blur_ratio",
        curve_param="filter_order",
        fixed_params={
            "bit_width_px": 6,
            "noise_sigma_adu": 300,
        },
        param_grid=param_grid,
    )

    plot_quality_heatmap(
        stats_df=stats_df,
        x_param="noise_sigma_adu",
        y_param="sigma_blur_ratio",
        fixed_params={
            "bit_width_px": 6.2,
            "filter_order": 2,
        },
        value_col="sigma_mean",
        param_grid=param_grid,
    )

    plt.show()