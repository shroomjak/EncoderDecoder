"""
calibrate_zebra.py — Estimate 1-D radial distortion coefficient k from a
zebra-stripe calibration signal (alternating equal-width bands).

Public API
----------
detect_zebra_edges             — find stripe edges with Blais-Rioux detector
estimate_k_from_zebra_signal   — optimise k by minimising inter-edge spacing variance
estimate_k_from_csv_line       — convenience wrapper for a CSV text line
estimate_k_from_serial         — convenience wrapper reading from a serial port
plot_zebra_calibration         — matplotlib diagnostic plot

"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from src.blais_rioux import (
    BRConfig, detect_edges,
)

from src.distortion.math1d import (
    parse_csv_signal_line,
    read_csv_signal_from_port,
    pixel_to_norm,
    norm_to_pixel, undistort_division_1d,
)

# ---------------------------------------------------------------------------
# k estimation
# ---------------------------------------------------------------------------

def _spacing_cost(k: float, edges_px: np.ndarray, n_pixels: int) -> float:
    """Variance-to-mean² of undistorted inter-edge spacings (minimised at k_opt).

    Вычисляется в пиксельных единицах undistorted-пространства.
    Математически эквивалентно вычислению в нормированных единицах
    (diff(x_u_px) = scale * diff(x_u_norm), scale сокращается),
    но явное использование пикселей делает смысл прозрачным и согласует
    масштаб с остальными метриками (period_px, residual_px).
    """
    x_u_norm = undistort_division_1d(pixel_to_norm(edges_px, n_pixels), k)
    if not np.all(np.isfinite(x_u_norm)):
        return np.inf
    x_u_px = norm_to_pixel(x_u_norm, n_pixels)
    d = np.diff(x_u_px)
    if len(d) < 2:
        return np.inf
    m = np.mean(d)
    return float(np.inf if abs(m) < 1e-15 else np.var(d) / (m ** 2 + 1e-15))


def estimate_k_from_zebra_signal(
    signal: np.ndarray,
    br_config: BRConfig,
    k_bounds: tuple[float, float] = (-0.24, 0.24),   # FIX: было (0.0, 0.24)
    return_debug: bool = False,
):
    """Estimate distortion coefficient k from a single zebra-stripe row.

    Parameters
    ----------
    signal : array
        Raw ADC pixel values.
    br_config : BRConfig
        Blais-Rioux detector configuration (bit_width_px = expected stripe width).
    k_bounds : (k_min, k_max)
        Search bounds for k.
        FIX: нижняя граница изменена с 0.0 на -0.24, чтобы охватывать
        подушкообразную дисторсию (k < 0). При старом значении 0.0
        minimize_scalar молча возвращал k = 0.0 для любого pincushion-объектива.
    return_debug : bool
        If True, return (k, debug_dict).

    Returns
    -------
    k if return_debug is False
    (k, debug_dict) if return_debug is True
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    detection_result = detect_edges(signal, br_config)
    edges_px = np.array([e.position for e in detection_result.detected_edges if e.d1_value > 0], dtype=np.float64)

    if len(edges_px) < 4:
        raise RuntimeError(
            "Too few zebra edges detected. "
            "Adjust BRConfig (bit_width_px, peak_threshold_rel, etc.)."
        )

    res = minimize_scalar(
        _spacing_cost, bounds=k_bounds, method="bounded", args=(edges_px, n)
    )
    k = float(res.x)

    if not return_debug:
        return k

    x_u_norm = undistort_division_1d(pixel_to_norm(edges_px, n), k)
    x_u_px   = norm_to_pixel(x_u_norm, n)
    idx      = np.arange(len(x_u_px), dtype=np.float64)
    a_px, b_px = np.polyfit(idx, x_u_px, 1)
    residual_px = x_u_px - (a_px * idx + b_px)

    debug = {
        "edges_px":             edges_px,
        "edges_undistorted_px": x_u_px,
        "residual_px":          residual_px,
        "rms_px":               float(np.sqrt(np.mean(residual_px ** 2))),
        "period_px":            float(a_px),
        "d1":                   detection_result.first_derivative,
        "d2":                   detection_result.second_derivative,
        "vignette_norm":    detection_result.vignette_norm,
        "peak_threshold":       detection_result.peak_threshold,
        "objective":            float(res.fun),
        "success":              bool(res.success),
    }
    return k, debug


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def estimate_k_from_csv_line(
    line: str,
    br_config: BRConfig,
    k_bounds: tuple[float, float] = (-0.24, 0.24),   # FIX: было (0.0, 0.24)
    return_debug: bool = True,
):
    row_idx, signal = parse_csv_signal_line(line)
    result = estimate_k_from_zebra_signal(signal, br_config, k_bounds, return_debug)
    if not return_debug:
        return row_idx, result
    k, debug = result
    debug["row_index"] = row_idx
    return row_idx, k, signal, debug


def estimate_k_from_serial(
    port: str,
    br_config: BRConfig,
    baudrate: int,
    timeout: float = 1.0,
    k_bounds: tuple[float, float] = (-0.24, 0.24),   # FIX: было (0.0, 0.24)
    return_debug: bool = True,
):
    row_idx, signal = read_csv_signal_from_port(port, baudrate, timeout)
    result = estimate_k_from_zebra_signal(signal, br_config, k_bounds, return_debug)
    if not return_debug:
        return row_idx, result
    k, debug = result
    debug["row_index"] = row_idx
    return row_idx, k, signal, debug


# ---------------------------------------------------------------------------
# Diagnostic plot (matplotlib, non-interactive use)
# ---------------------------------------------------------------------------

def plot_zebra_calibration(signal: np.ndarray, debug: dict) -> None:
    """Three-panel diagnostic: raw signal + edges, spacing comparison, residuals."""
    signal      = np.asarray(signal, dtype=np.float64)
    edges_px    = debug["edges_px"]
    edges_undist = debug["edges_undistorted_px"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(signal, color="black", lw=1.0, label="ADC signal")
    for x in edges_px:
        axs[0].axvline(x, color="red", alpha=0.2)
    axs[0].set_title("Zebra signal and detected edges")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(np.diff(edges_px),    "o-", color="red",   label="Before correction")
    axs[1].plot(np.diff(edges_undist), "x-", color="green", label="After correction")
    axs[1].set_title(f"Inter-edge spacings T_mean = {np.diff(edges_undist).mean():.2f} m")
    axs[1].set_ylabel("px")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(debug["residual_px"], "o-", color="purple")
    axs[2].axhline(0.0, color="gray", lw=1.0)
    axs[2].set_title(f"Residuals RMS = {debug['rms_px']:.4f} px")
    axs[2].set_xlabel("Edge index")
    axs[2].set_ylabel("px")
    axs[2].grid(True)

    plt.tight_layout()

if __name__ == "__main__":

    _, k1, _, debug = estimate_k_from_csv_line(
        line="1,0.249211356466877,0.570977917981073,7.23343848580442,71.3059936908517,477.495268138801,1132.31861198738,1291.86750788644,1375.6309148265,1455.77287066246,1567.34700315457,1241.28075709779,196.798107255521,8.60883280757098,3.18927444794953,8.87066246056782,7.5205047318612,17.8706624605678,31.6246056782334,140.94952681388,2007.6403785489,2547.52365930599,2645.51735015773,2665.80757097792,2736.34069400631,2779.94321766562,2855.04100946372,771.252365930599,22.4542586750789,15.7350157728707,12.5047318611987,18.186119873817,21.2334384858044,33.8864353312303,61.9400630914827,1337.77287066246,3347.93375394322,3532.93059936909,3525.69085173502,3567.27129337539,3603.43848580442,3632.1167192429,3568.74132492114,596.97476340694,45.391167192429,40.0883280757098,38.7066246056782,43.7981072555205,47.1955835962145,51.608832807571,77.9116719242902,1343.43848580442,2958.97476340694,2945.82018927445,2931.3596214511,2906.36277602524,2899.92113564669,2798.5141955836,2814.03785488959,1041.72555205047,44.0883280757098,26.9905362776025,24.3280757097792,23.217665615142,19.0946372239748,22.0914826498423,29.1388012618297,872.167192429022,2954.78548895899,2936.2334384858,2961.06940063092,2941.41640378549,2959.79495268139,2915.51104100946,2832.19558359621,255.053627760252,30.2365930599369,17.5488958990536,10.3343848580442,11.820189274448,7.69716088328076,11.9085173501577,26.5930599369085,1794.82649842271,2638.94952681388,2668.72239747634,2643.14195583596,2618.1356466877,2599.84858044164,2397.98107255521,445.375394321767,44.2145110410095,13.2145110410095,8.64668769716088,2.29652996845426,2.8391167192429,0.640378548895899,48.205047318612,1111.27129337539,1744.33123028391,1724.57413249211,1676.261829653,1633.32807570978,1444.53943217666,639.293375394322,81.5804416403786,5.39747634069401,0.905362776025237,0.391167192429022",
        br_config=BRConfig(
            bit_width_px=7.2,
        ),
    )
    print(k1)
    plot_zebra_calibration(debug["vignette_norm"], debug)

    _, k2, _, debug = estimate_k_from_csv_line(
        line = """
        2,501.811083123426,804.403022670025,920.43073047859,1009.93198992443,1092.59445843829,973.065491183879,401.705289672544,4.70025188916877,1.41057934508816,1.68261964735516,6.73551637279597,7.46599496221663,18.5113350125945,83.0251889168766,1046.55667506297,2092.50881612091,2219.87405541562,2360.71032745592,2359.5717884131,2521.05541561713,2494.31486146096,1117.27204030227,19.816120906801,9.74307304785894,14.0982367758186,14.5390428211587,21.3123425692695,31.3476070528967,64.0579345088161,1216.97229219144,3160.28715365239,3335.19143576826,3190.5516372796,3379.59697732998,3372.2443324937,3381.54911838791,3453.00755667506,724.219143576826,36.1486146095718,22.992443324937,25.7732997481108,22.7128463476071,30.624685138539,35.8589420654912,66.0654911838791,1058.42317380353,3115.45088161209,3115.70025188917,3107.99748110831,3062.03526448363,3024.03778337532,2998.78337531486,2918.16120906801,1041.40554156171,67.55919395466,42.7758186397985,38.0025188916877,28.0755667506297,29.1486146095718,28.4156171284635,41.9269521410579,465.566750629723,2410.30982367758,2803.76070528967,2934.21410579345,2987.89168765743,2956.31989924433,2999.69521410579,2922.75314861461,1102.07304785894,43.3576826196474,19.3173803526448,16.0025188916877,11.4307304785894,12.4987405541562,9.05793450881612,19.1561712846348,679.816120906801,2761.40302267003,2747.18136020151,2780.05793450882,2725.3274559194,2696.41057934509,2675.15617128463,1783.68765743073,104.385390428212,27.183879093199,14.2745591939547,11.7682619647355,4.58438287153652,6.6624685138539,4.18136020151134,297.410579345088,1935.02770780856,2225.50377833753,2192.94458438287,2128.56675062972,1954.49874055416,1608.76574307305,462.725440806045,44.3702770780856,6.62468513853904,2.86649874055416,1.03778337531486,0.884130982367758,1.05541561712846,159.493702770781,664.828715365239
        """,
        br_config=BRConfig(
            bit_width_px=12,
        ),
    )
    print(k2)
    plot_zebra_calibration(debug["vignette_norm"], debug)

    _, k3, _, debug = estimate_k_from_csv_line(
        line="""
            3,211.013282732448,612.671726755218,889.370018975332,985.195445920304,1077.09297912713,1147.44781783681,966.713472485769,280.721062618596,1.07020872865275,0.658444022770399,3.1404174573055,1.65085388994307,6.42504743833017,15.1157495256167,146.368121442125,1515.78368121442,2198.37760910816,2340.65654648956,2333.65085388994,2487.49335863378,2481.65654648956,2366.98292220114,314.476280834915,10.1973434535104,9.29601518026566,6.75901328273245,12.8690702087287,15.9392789373814,28.8557874762808,73.753320683112,1993.59962049336,3151.13472485769,3034.55787476281,3198.0284629981,3188.86527514232,3194.65085388994,3295.43074003795,3039.57115749526,113.032258064516,20.9089184060721,18.9696394686907,17.8690702087287,20.8007590132827,23.0417457305503,40.438330170778,109.713472485769,2198.09487666034,3015.13282732448,3029.67172675522,3005.28652751423,2981.5275142315,2967.18595825427,2922.92599620493,2754.63377609108,459.370018975332,45.6413662239089,33.5977229601518,21.1745730550285,22.9506641366224,20.303605313093,25.6527514231499,42.5142314990512,1202.51992409867,2765.95825426945,2907.97722960152,2961.92409867173,2934.30360531309,2980.21252371917,2919.92789373814,2897.24478178368,500.848197343454,30.3946869070209,16.0645161290323,6.5180265654649,7.41555977229602,3.45540796963947,7.04174573055029,17.2637571157495,1328.4440227704,2716.06641366224,2768.49335863378,2717.68880455408,2689.53889943074,2674.12333965844,2660.45730550285,800.986717267552,54.2239089184061,15.4098671726755,9.46869070208729,2.35673624288425,2.36622390891841,0.755218216318786,2.98481973434535,577.314990512334,1888.22770398482,2001.17647058824,1943.889943074,1776.36812144213,1615.82352941176,1213.3605313093,276.049335863378,20.9639468690702,2.19544592030361,0.239089184060721,1.00948766603416,0.586337760910816,0.962049335863378,226.286527514232
            """,
        br_config=BRConfig(
            bit_width_px=12,
        ),
    )
    print(k3)
    plot_zebra_calibration(debug["vignette_norm"], debug)
    plt.show()

