#!/usr/bin/env python3
"""Создаёт графики распределений с разной параметризацией"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # позволяет запускать в CI и сборках документации без дисплея
import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats
from tqdm import tqdm

Array = np.ndarray
Pdf = Callable[[Array], Array]

# Общие параметры оформления и рендеринга графиков.
COLORS = ("#e41a1c", "#ff9800", "#00c853", "#222222", "#2962ff", "#d500f9")
FIGURE_SIZE = (5, 4)
FIGURE_LAYOUT = "constrained"
DEFAULT_DPI = 180
FIGURE_FACE_COLOR = "#f7f7f7"
CURVE_SAMPLES = 1_500

TITLE_FONT_SIZE = 15
TITLE_PADDING = 12
LABEL_FONT_SIZE = 10
X_AXIS_LABEL = "y"
CONTINUOUS_Y_AXIS_LABEL = "probability density"
DISCRETE_Y_AXIS_LABEL = "probability mass"
MIXED_Y_AXIS_LABEL = "density / endpoint probability mass"

CURVE_LINE_WIDTH = 2
STEM_LINE_WIDTH = 1.4
MASS_LINE_WIDTH = 1.5
STEM_ALPHA = 0.8
MASS_ALPHA = 0.8
MASS_LINE_STYLE = ":"
MARKER_STYLE = "o"
DISCRETE_MARKER_SIZE = 4
MASS_MARKER_SIZE = 5

GRID_COLOR = "#9e9e9e"
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 0.7
GRID_ALPHA = 0.45
HIDDEN_SPINES = ("top", "right")

DISCRETE_X_MARGIN = 0.6
DISCRETE_TICK_COUNT = 10
MASS_ENDPOINTS = (0.0, 1.0)

LEGEND_STYLE = {
    "loc": "upper right",
    "fontsize": LABEL_FONT_SIZE,
    "frameon": True,
    "facecolor": "white",
    "framealpha": 0.88,
    "edgecolor": "#bdbdbd",
}


@dataclass(frozen=True)
class Curve:
    label: str
    pdf: Pdf


@dataclass(frozen=True)
class ContinuousChart:
    slug: str
    title: str
    xlim: tuple[float, float]
    curves: tuple[Curve, ...]


@dataclass(frozen=True)
class DiscreteChart:
    slug: str
    title: str
    xmax: int
    curves: tuple[tuple[str, Callable[[Array], Array]], ...]


@dataclass(frozen=True)
class MixedCurve:
    """Плотность на открытом интервале с массами вероятности на его концах."""

    label: str
    pdf: Pdf
    masses: tuple[float, float]


@dataclass(frozen=True)
class MixedChart:
    slug: str
    title: str
    xlim: tuple[float, float]
    curves: tuple[MixedCurve, ...]


def gamma_mean_cv(mean: float, cv: float) -> Curve:
    shape, rate = cv**-2, 1 / (mean * cv**2)
    return Curve(
        rf"$\mu={mean:g},\ c={cv:g}$",
        lambda x: stats.gamma.pdf(x, shape, scale=1 / rate),
    )


def inverse_gaussian(mean: float, shape: float) -> Curve:
    def pdf(x: Array) -> Array:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.sqrt(shape / (2 * np.pi * x**3)) * np.exp(
                -shape * (x - mean) ** 2 / (2 * mean**2 * x)
            )
        return np.where(x > 0, result, 0.0)

    return Curve(rf"$\mu={mean:g},\ \lambda={shape:g}$", pdf)


def generalized_gamma(scale: float, sigma: float, nu: float) -> Curve:
    """GAMLSS GG: y=scale*(Gamma(k,1))**(sigma*nu), k=1/(sigma²nu²)."""
    k = 1 / (sigma * sigma * nu * nu)
    power = sigma * nu

    def pdf(x: Array) -> Array:
        z = x / scale
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = np.abs(1 / power) * z ** (1 / power - 1) * np.exp(-(z ** (1 / power)))
            out /= scale * special.gamma(k)
            out *= (z ** (1 / power)) ** (k - 1)  # гамма-ядро после преобразования
        return np.where(x > 0, out, 0.0)

    return Curve(rf"$\mu={scale:g},\ \sigma={sigma:g},\ \nu={nu:g}$", pdf)


def power_exponential(mu: float, sigma: float, nu: float) -> Curve:
    # sigma — стандартное отклонение в gamlss-family, а не исходный масштаб GED.
    a = sigma * np.sqrt(special.gamma(1 / nu) / special.gamma(3 / nu))
    return Curve(
        rf"$\mu={mu:g},\ \sigma={sigma:g},\ \nu={nu:g}$",
        lambda x: stats.gennorm.pdf(x, nu, loc=mu, scale=a),
    )


def shash(mu: float, sigma: float, nu: float, tau: float) -> Curve:
    def pdf(x: Array) -> Array:
        z = (x - mu) / sigma
        h = tau * np.arcsinh(z) - np.log(nu)
        return stats.norm.pdf(h) * tau / (sigma * np.sqrt(1 + z * z))

    return Curve(rf"$\mu={mu:g},\ \sigma={sigma:g},\ \nu={nu:g},\ \tau={tau:g}$", pdf)


def skew_t(mu: float, sigma: float, nu: float, tau: float) -> Curve:
    def pdf(x: Array) -> Array:
        z = (x - mu) / sigma
        arg = nu * z * np.sqrt((tau + 1) / (tau + z * z))
        return 2 * stats.t.pdf(z, tau) * stats.t.cdf(arg, tau + 1) / sigma

    return Curve(rf"$\mu={mu:g},\ \sigma={sigma:g},\ \nu={nu:g},\ \tau={tau:g}$", pdf)


def zaga(mean: float, cv: float, zero: float) -> tuple[str, Callable[[Array], Array]]:
    shape, rate = cv**-2, 1 / (mean * cv**2)
    return (
        rf"$\mu={mean:g},\ c={cv:g},\ \pi_0={zero:g}$",
        lambda k: np.where(
            k == 0, zero, (1 - zero) * stats.gamma.pdf(k, shape, scale=1 / rate)
        ),
    )


def beinf(
    mu: float, sigma: float, nu: float, tau: float
) -> tuple[str, Callable[[Array], Array]]:
    # Строится непрерывная бета-часть; массы на концах отмечаются на осях.
    precision = 1 / sigma - 1
    alpha, beta = mu * precision, (1 - mu) * precision
    normalizer = 1 + nu + tau
    return (
        rf"$\mu={mu:g},\ \sigma={sigma:g},\ \nu={nu:g},\ \tau={tau:g}$",
        lambda x: stats.beta.pdf(x, alpha, beta) / normalizer,
    )


def gamma_mean_shape(mean: float, shape: float) -> Curve:
    return Curve(
        rf"$\mu={mean:g},\ \alpha={shape:g}$",
        lambda x: stats.gamma.pdf(x, shape, scale=mean / shape),
    )


def inverse_gaussian_mean_cv(mean: float, cv: float) -> Curve:
    curve = inverse_gaussian(mean, mean / cv**2)
    return Curve(rf"$\mu={mean:g},\ c={cv:g}$", curve.pdf)


def weibull_mean_shape(mean: float, shape: float) -> Curve:
    scale = mean / special.gamma(1 + 1 / shape)
    return Curve(
        rf"$\mu={mean:g},\ k={shape:g}$",
        lambda x: stats.weibull_min.pdf(x, shape, scale=scale),
    )


def log_normal_mean_log_sd(mean: float, log_sd: float) -> Curve:
    location = np.log(mean) - log_sd**2 / 2
    return Curve(
        rf"$\mu={mean:g},\ s={log_sd:g}$",
        lambda x: stats.lognorm.pdf(x, log_sd, scale=np.exp(location)),
    )


def log_normal_mean_cv(mean: float, cv: float) -> Curve:
    log_sd = np.sqrt(np.log1p(cv**2))
    curve = log_normal_mean_log_sd(mean, log_sd)
    return Curve(rf"$\mu={mean:g},\ c={cv:g}$", curve.pdf)


def log_normal_median_log_sd(median: float, log_sd: float) -> Curve:
    return Curve(
        rf"$q_{{0.5}}={median:g},\ s={log_sd:g}$",
        lambda x: stats.lognorm.pdf(x, log_sd, scale=median),
    )


def skew_normal_mean_sd(mean: float, sd: float, nu: float) -> Curve:
    delta = nu / np.hypot(nu, 1)
    standardized_mean = np.sqrt(2 / np.pi) * delta
    scale = sd / np.sqrt(1 - standardized_mean**2)
    location = mean - scale * standardized_mean
    return Curve(
        rf"$m={mean:g},\ s={sd:g},\ \nu={nu:g}$",
        lambda x: stats.skewnorm.pdf(x, nu, loc=location, scale=scale),
    )


def skew_t_mean_sd(mean: float, sd: float, nu: float, tau: float) -> Curve:
    delta = nu / np.hypot(nu, 1)
    standardized_mean = (
        delta
        * np.sqrt(tau / np.pi)
        * np.exp(special.gammaln((tau - 1) / 2) - special.gammaln(tau / 2))
    )
    standardized_variance = 1 + 2 / (tau - 2) - standardized_mean**2
    scale = sd / np.sqrt(standardized_variance)
    curve = skew_t(mean - scale * standardized_mean, scale, nu, tau)
    return Curve(rf"$m={mean:g},\ s={sd:g},\ \nu={nu:g},\ \tau={tau:g}$", curve.pdf)


def tweedie(mean: float, dispersion: float, power: float) -> MixedCurve:
    """Составная пуассоновско-гамма-плотность для gamlss-family при 1 < p < 2."""
    lam = mean ** (2 - power) / (dispersion * (2 - power))
    alpha = (2 - power) / (power - 1)
    scale = dispersion * (power - 1) * mean ** (power - 1)

    def pdf(x: Array) -> Array:
        result = np.zeros_like(x, dtype=float)
        positive = x > 0
        y = x[positive]
        log_sum = np.full_like(y, -np.inf, dtype=float)
        # Пуассоновским хвостом можно пренебречь при этих параметрах для документации.
        for n in range(1, 250):
            shape = n * alpha
            term = -lam + n * np.log(lam) - special.gammaln(n + 1)
            term += stats.gamma.logpdf(y, shape, scale=scale)
            log_sum = np.logaddexp(log_sum, term)
        result[positive] = np.exp(log_sum)
        return result

    return MixedCurve(
        rf"$\mu={mean:g},\ \phi={dispersion:g},\ p={power:g}$", pdf, (np.exp(-lam), 0.0)
    )


def tweedie_mean_cv(mean: float, cv: float, power: float) -> MixedCurve:
    curve = tweedie(mean, cv**2 * mean ** (2 - power), power)
    return MixedCurve(
        rf"$\mu={mean:g},\ c={cv:g},\ p={power:g}$", curve.pdf, curve.masses
    )


def beinf_curve(mu: float, sigma: float, nu: float, tau: float) -> MixedCurve:
    precision = 1 / sigma - 1
    alpha, beta = mu * precision, (1 - mu) * precision
    denominator = 1 + nu + tau
    return MixedCurve(
        rf"$\mu={mu:g},\ \sigma={sigma:g},\ \nu={nu:g},\ \tau={tau:g}$",
        lambda x: stats.beta.pdf(x, alpha, beta) / denominator,
        (nu / denominator, tau / denominator),
    )


def zaga_curve(
    component_mean: float, cv: float, zero: float, *, total: bool = False
) -> MixedCurve:
    mean = component_mean / (1 - zero) if total else component_mean
    shape, rate = cv**-2, 1 / (mean * cv**2)
    first = "m" if total else r"\mu"
    return MixedCurve(
        rf"${first}={component_mean:g},\ c={cv:g},\ \pi_0={zero:g}$",
        lambda x: (1 - zero) * stats.gamma.pdf(x, shape, scale=1 / rate),
        (zero, 0.0),
    )


CONTINUOUS: tuple[ContinuousChart, ...] = (
    ContinuousChart(
        "normal",
        "Normal distribution",
        (-5, 5),
        (
            Curve(r"$\mu=0,\ \sigma=0.5$", lambda x: stats.norm.pdf(x, 0, 0.5)),
            Curve(r"$\mu=0,\ \sigma=1$", lambda x: stats.norm.pdf(x)),
            Curve(r"$\mu=1.5,\ \sigma=1$", lambda x: stats.norm.pdf(x, 1.5)),
        ),
    ),
    ContinuousChart(
        "gamma",
        "Gamma distribution (shape/rate)",
        (0, 14),
        (
            Curve(r"$\alpha=1.2,\ \beta=1$", lambda x: stats.gamma.pdf(x, 1.2)),
            Curve(r"$\alpha=2,\ \beta=1$", lambda x: stats.gamma.pdf(x, 2)),
            Curve(r"$\alpha=5,\ \beta=1$", lambda x: stats.gamma.pdf(x, 5)),
            Curve(r"$\alpha=5,\ \beta=2$", lambda x: stats.gamma.pdf(x, 5, scale=0.5)),
        ),
    ),
    ContinuousChart(
        "gamma_mean_cv",
        "Gamma distribution (mean/CV)",
        (0, 14),
        tuple(gamma_mean_cv(*p) for p in ((2, 0.35), (4, 0.5), (6, 0.7))),
    ),
    ContinuousChart(
        "gamma_mean_shape",
        "Gamma distribution (mean/shape)",
        (0, 14),
        tuple(gamma_mean_shape(*p) for p in ((2, 1.5), (4, 4), (6, 8))),
    ),
    ContinuousChart(
        "beta",
        "Beta distribution (mean/precision)",
        (0.001, 0.999),
        (
            Curve(r"$\mu=0.5,\ \phi=4$", lambda x: stats.beta.pdf(x, 2, 2)),
            Curve(r"$\mu=0.5,\ \phi=12$", lambda x: stats.beta.pdf(x, 6, 6)),
            Curve(r"$\mu=0.25,\ \phi=12$", lambda x: stats.beta.pdf(x, 3, 9)),
            Curve(r"$\mu=0.75,\ \phi=12$", lambda x: stats.beta.pdf(x, 9, 3)),
        ),
    ),
    ContinuousChart(
        "exponential",
        "Exponential distribution",
        (0, 8),
        (
            Curve(r"$\lambda=0.5$", lambda x: stats.expon.pdf(x, scale=2)),
            Curve(r"$\lambda=1$", stats.expon.pdf),
            Curve(r"$\lambda=2$", lambda x: stats.expon.pdf(x, scale=0.5)),
        ),
    ),
    ContinuousChart(
        "exponential_mean",
        "Exponential distribution (mean)",
        (0, 10),
        (
            Curve(r"$\mu=0.75$", lambda x: stats.expon.pdf(x, scale=0.75)),
            Curve(r"$\mu=2$", lambda x: stats.expon.pdf(x, scale=2)),
            Curve(r"$\mu=4$", lambda x: stats.expon.pdf(x, scale=4)),
        ),
    ),
    ContinuousChart(
        "log_normal",
        "Log-normal distribution",
        (0.001, 12),
        (
            Curve(r"$m=0,\ s=0.35$", lambda x: stats.lognorm.pdf(x, 0.35, scale=1)),
            Curve(r"$m=0,\ s=0.8$", lambda x: stats.lognorm.pdf(x, 0.8, scale=1)),
            Curve(r"$m=1,\ s=0.5$", lambda x: stats.lognorm.pdf(x, 0.5, scale=np.e)),
        ),
    ),
    ContinuousChart(
        "log_normal_mean_log_sd",
        "Log-normal distribution (mean/log-SD)",
        (0.001, 14),
        tuple(log_normal_mean_log_sd(*p) for p in ((2, 0.35), (4, 0.7), (6, 1))),
    ),
    ContinuousChart(
        "log_normal_mean_cv",
        "Log-normal distribution (mean/CV)",
        (0.001, 18),
        tuple(log_normal_mean_cv(*p) for p in ((2, 0.3), (4, 0.7), (6, 1.2))),
    ),
    ContinuousChart(
        "log_normal_median_log_sd",
        "Log-normal distribution (median/log-SD)",
        (0.001, 14),
        tuple(log_normal_median_log_sd(*p) for p in ((2, 0.35), (4, 0.7), (6, 1))),
    ),
    ContinuousChart(
        "inverse_gaussian",
        "Inverse Gaussian distribution (mean/shape)",
        (0.001, 12),
        tuple(inverse_gaussian(*p) for p in ((1, 1), (2, 2), (3, 8))),
    ),
    ContinuousChart(
        "inverse_gaussian_mean_cv",
        "Inverse Gaussian distribution (mean/CV)",
        (0.001, 18),
        tuple(inverse_gaussian_mean_cv(*p) for p in ((1, 0.8), (2, 0.5), (3, 0.35))),
    ),
    ContinuousChart(
        "weibull",
        "Weibull distribution (scale/shape)",
        (0, 8),
        (
            Curve(r"$a=1,\ k=1.2$", lambda x: stats.weibull_min.pdf(x, 1.2)),
            Curve(r"$a=1,\ k=1$", lambda x: stats.weibull_min.pdf(x, 1)),
            Curve(r"$a=2,\ k=2$", lambda x: stats.weibull_min.pdf(x, 2, scale=2)),
            Curve(r"$a=3,\ k=4$", lambda x: stats.weibull_min.pdf(x, 4, scale=3)),
        ),
    ),
    ContinuousChart(
        "weibull_mean_shape",
        "Weibull distribution (mean/shape)",
        (0, 10),
        tuple(weibull_mean_shape(*p) for p in ((2, 1), (3, 2), (4, 4))),
    ),
    ContinuousChart(
        "lomax",
        "Lomax distribution",
        (0, 15),
        (
            Curve(
                r"$\alpha=0.8,\ \lambda=2$", lambda x: stats.lomax.pdf(x, 0.8, scale=2)
            ),
            Curve(r"$\alpha=2,\ \lambda=2$", lambda x: stats.lomax.pdf(x, 2, scale=2)),
            Curve(r"$\alpha=5,\ \lambda=2$", lambda x: stats.lomax.pdf(x, 5, scale=2)),
        ),
    ),
    ContinuousChart(
        "laplace",
        "Laplace distribution",
        (-6, 6),
        (
            Curve(r"$\mu=0,\ \sigma=0.5$", lambda x: stats.laplace.pdf(x, scale=0.5)),
            Curve(r"$\mu=0,\ \sigma=1$", stats.laplace.pdf),
            Curve(r"$\mu=2,\ \sigma=1$", lambda x: stats.laplace.pdf(x, 2)),
        ),
    ),
    ContinuousChart(
        "logistic",
        "Logistic distribution",
        (-8, 8),
        (
            Curve(r"$\mu=0,\ \sigma=0.7$", lambda x: stats.logistic.pdf(x, scale=0.7)),
            Curve(r"$\mu=0,\ \sigma=1.5$", lambda x: stats.logistic.pdf(x, scale=1.5)),
            Curve(r"$\mu=2,\ \sigma=1$", lambda x: stats.logistic.pdf(x, 2)),
        ),
    ),
    ContinuousChart(
        "gumbel",
        "Gumbel distribution",
        (-5, 10),
        (
            Curve(r"$\mu=0,\ \sigma=1$", stats.gumbel_r.pdf),
            Curve(r"$\mu=0,\ \sigma=2$", lambda x: stats.gumbel_r.pdf(x, scale=2)),
            Curve(r"$\mu=3,\ \sigma=1$", lambda x: stats.gumbel_r.pdf(x, 3)),
        ),
    ),
    ContinuousChart(
        "gev",
        "Generalized extreme-value distribution",
        (-5, 8),
        (
            Curve(
                r"$\mu=0,\ \sigma=1,\ \nu=-0.4$",
                lambda x: stats.genextreme.pdf(x, 0.4),
            ),
            Curve(r"$\mu=0,\ \sigma=1,\ \nu=0$", stats.gumbel_r.pdf),
            Curve(
                r"$\mu=0,\ \sigma=1,\ \nu=0.4$",
                lambda x: stats.genextreme.pdf(x, -0.4),
            ),
        ),
    ),
    ContinuousChart(
        "student_t",
        "Student's t distribution (location/scale/DF)",
        (-6, 6),
        (
            Curve(r"$\mu=0,\ \sigma=1,\ \tau=1$", lambda x: stats.t.pdf(x, 1)),
            Curve(r"$\mu=0,\ \sigma=1,\ \tau=2$", lambda x: stats.t.pdf(x, 2)),
            Curve(r"$\mu=0,\ \sigma=1,\ \tau=5$", lambda x: stats.t.pdf(x, 5)),
            Curve(r"$\mu=0,\ \sigma=1,\ \tau=30$", lambda x: stats.t.pdf(x, 30)),
        ),
    ),
    ContinuousChart(
        "student_t_fixed",
        "Student's t distribution (fixed DF)",
        (-8, 8),
        (
            Curve(r"$\mu=0,\ \sigma=1,\ \tau=5$", lambda x: stats.t.pdf(x, 5)),
            Curve(r"$\mu=2,\ \sigma=1,\ \tau=5$", lambda x: stats.t.pdf(x, 5, loc=2)),
            Curve(r"$\mu=0,\ \sigma=2,\ \tau=5$", lambda x: stats.t.pdf(x, 5, scale=2)),
        ),
    ),
    ContinuousChart(
        "student_t_stddev",
        "Student's t distribution (mean/SD/DF)",
        (-8, 8),
        (
            Curve(
                r"$\mu=0,\ s=1,\ \tau=3$",
                lambda x: stats.t.pdf(x, 3, scale=np.sqrt(1 / 3)),
            ),
            Curve(
                r"$\mu=0,\ s=1,\ \tau=8$",
                lambda x: stats.t.pdf(x, 8, scale=np.sqrt(6 / 8)),
            ),
            Curve(
                r"$\mu=2,\ s=2,\ \tau=8$",
                lambda x: stats.t.pdf(x, 8, loc=2, scale=2 * np.sqrt(6 / 8)),
            ),
        ),
    ),
    ContinuousChart(
        "skew_normal",
        "Skew-normal distribution",
        (-5, 5),
        (
            Curve(r"$\mu=0,\ \sigma=1,\ \nu=-4$", lambda x: stats.skewnorm.pdf(x, -4)),
            Curve(r"$\mu=0,\ \sigma=1,\ \nu=0$", lambda x: stats.skewnorm.pdf(x, 0)),
            Curve(r"$\mu=0,\ \sigma=1,\ \nu=4$", lambda x: stats.skewnorm.pdf(x, 4)),
        ),
    ),
    ContinuousChart(
        "skew_normal_mean_sd",
        "Skew-normal distribution (mean/SD)",
        (-5, 5),
        tuple(skew_normal_mean_sd(*p) for p in ((0, 1, -4), (0, 1, 0), (0, 1, 4))),
    ),
    ContinuousChart(
        "skew_student_t",
        "Skew Student's t distribution",
        (-8, 8),
        tuple(
            skew_t(*p)
            for p in ((0, 1, -3, 4), (0, 1, 0, 4), (0, 1, 3, 4), (0, 1, 3, 15))
        ),
    ),
    ContinuousChart(
        "skew_student_t_mean_sd",
        "Skew Student's t distribution (mean/SD)",
        (-8, 8),
        tuple(skew_t_mean_sd(*p) for p in ((0, 1, -3, 5), (0, 1, 0, 5), (0, 1, 3, 5))),
    ),
    ContinuousChart(
        "power_exponential",
        "Power exponential distribution",
        (-5, 5),
        tuple(power_exponential(*p) for p in ((0, 1, 1), (0, 1, 2), (0, 1, 4))),
    ),
    ContinuousChart(
        "generalized_gamma",
        "Generalized gamma distribution",
        (0.001, 12),
        tuple(
            generalized_gamma(*p) for p in ((2, 0.7, 0.7), (2, 0.7, 1), (2, 0.7, 1.25))
        ),
    ),
    ContinuousChart(
        "johnson_su",
        "Johnson SU distribution",
        (-8, 8),
        (
            Curve(
                r"$\mu=0,\ \sigma=1,\ \nu=0,\ \tau=1$",
                lambda x: stats.johnsonsu.pdf(x, 0, 1),
            ),
            Curve(
                r"$\mu=0,\ \sigma=1,\ \nu=2,\ \tau=1$",
                lambda x: stats.johnsonsu.pdf(x, 2, 1),
            ),
            Curve(
                r"$\mu=0,\ \sigma=1,\ \nu=0,\ \tau=2$",
                lambda x: stats.johnsonsu.pdf(x, 0, 2),
            ),
        ),
    ),
    ContinuousChart(
        "shash",
        "Sinh-arcsinh (SHASH) distribution",
        (-6, 6),
        tuple(
            shash(*p)
            for p in ((0, 1, 1, 1), (0, 1, 0.5, 1), (0, 1, 1, 0.6), (0, 1, 1, 1.7))
        ),
    ),
)

DISCRETE: tuple[DiscreteChart, ...] = (
    DiscreteChart(
        "bernoulli",
        "Bernoulli distribution",
        1,
        (
            (r"$p=0.2$", lambda k: stats.bernoulli.pmf(k, 0.2)),
            (r"$p=0.5$", lambda k: stats.bernoulli.pmf(k, 0.5)),
            (r"$p=0.8$", lambda k: stats.bernoulli.pmf(k, 0.8)),
        ),
    ),
    DiscreteChart(
        "poisson",
        "Poisson distribution",
        20,
        (
            (r"$\mu=2$", lambda k: stats.poisson.pmf(k, 2)),
            (r"$\mu=5$", lambda k: stats.poisson.pmf(k, 5)),
            (r"$\mu=10$", lambda k: stats.poisson.pmf(k, 10)),
        ),
    ),
    DiscreteChart(
        "negative_binomial",
        "Negative binomial distribution (mean/size)",
        30,
        (
            (r"$\mu=8,\ r=2$", lambda k: stats.nbinom.pmf(k, 2, 2 / (2 + 8))),
            (r"$\mu=8,\ r=8$", lambda k: stats.nbinom.pmf(k, 8, 8 / (8 + 8))),
            (r"$\mu=15,\ r=4$", lambda k: stats.nbinom.pmf(k, 4, 4 / (4 + 15))),
        ),
    ),
    DiscreteChart(
        "negative_binomial_mean_dispersion",
        "Negative binomial distribution (mean/dispersion)",
        30,
        (
            (r"$\mu=8,\ \phi=0.5$", lambda k: stats.nbinom.pmf(k, 2, 0.2)),
            (r"$\mu=8,\ \phi=0.125$", lambda k: stats.nbinom.pmf(k, 8, 0.5)),
            (r"$\mu=15,\ \phi=0.25$", lambda k: stats.nbinom.pmf(k, 4, 4 / 19)),
        ),
    ),
    DiscreteChart(
        "zip",
        "Zero-inflated Poisson distribution",
        20,
        (
            (
                r"$\lambda=6,\ \pi_0=0.1$",
                lambda k: np.where(
                    k == 0,
                    0.1 + 0.9 * stats.poisson.pmf(0, 6),
                    0.9 * stats.poisson.pmf(k, 6),
                ),
            ),
            (
                r"$\lambda=6,\ \pi_0=0.35$",
                lambda k: np.where(
                    k == 0,
                    0.35 + 0.65 * stats.poisson.pmf(0, 6),
                    0.65 * stats.poisson.pmf(k, 6),
                ),
            ),
        ),
    ),
    DiscreteChart(
        "zip_total_mean",
        "Zero-inflated Poisson distribution (total mean)",
        20,
        (
            (
                r"$m=4,\ \pi_0=0.2$",
                lambda k: np.where(
                    k == 0,
                    0.2 + 0.8 * stats.poisson.pmf(0, 5),
                    0.8 * stats.poisson.pmf(k, 5),
                ),
            ),
            (
                r"$m=4,\ \pi_0=0.5$",
                lambda k: np.where(
                    k == 0,
                    0.5 + 0.5 * stats.poisson.pmf(0, 8),
                    0.5 * stats.poisson.pmf(k, 8),
                ),
            ),
        ),
    ),
    DiscreteChart(
        "zinb",
        "Zero-inflated negative binomial distribution",
        30,
        (
            (
                r"$\mu=10,\ r=3,\ \pi_0=0.15$",
                lambda k: np.where(
                    k == 0,
                    0.15 + 0.85 * stats.nbinom.pmf(0, 3, 3 / 13),
                    0.85 * stats.nbinom.pmf(k, 3, 3 / 13),
                ),
            ),
            (
                r"$\mu=10,\ r=3,\ \pi_0=0.35$",
                lambda k: np.where(
                    k == 0,
                    0.35 + 0.65 * stats.nbinom.pmf(0, 3, 3 / 13),
                    0.65 * stats.nbinom.pmf(k, 3, 3 / 13),
                ),
            ),
        ),
    ),
    DiscreteChart(
        "zinb_total_mean",
        "Zero-inflated negative binomial distribution (total mean)",
        30,
        (
            (
                r"$m=6,\ r=3,\ \pi_0=0.2$",
                lambda k: np.where(
                    k == 0,
                    0.2 + 0.8 * stats.nbinom.pmf(0, 3, 3 / 10.5),
                    0.8 * stats.nbinom.pmf(k, 3, 3 / 10.5),
                ),
            ),
            (
                r"$m=6,\ r=3,\ \pi_0=0.5$",
                lambda k: np.where(
                    k == 0,
                    0.5 + 0.5 * stats.nbinom.pmf(0, 3, 0.2),
                    0.5 * stats.nbinom.pmf(k, 3, 0.2),
                ),
            ),
        ),
    ),
)


MIXED: tuple[MixedChart, ...] = (
    MixedChart(
        "tweedie_mean_dispersion_power",
        "Tweedie distribution (mean/dispersion/power)",
        (-0.4, 12),
        tuple(tweedie(*p) for p in ((2, 1, 1.3), (2, 1, 1.5), (2, 1, 1.7))),
    ),
    MixedChart(
        "tweedie_mean_cv_power",
        "Tweedie distribution (mean/CV/power)",
        (-0.4, 12),
        tuple(
            tweedie_mean_cv(*p) for p in ((2, 0.7, 1.3), (2, 0.7, 1.5), (2, 0.7, 1.7))
        ),
    ),
    MixedChart(
        "beinf",
        "Beta inflated at zero and one (BEINF)",
        (-0.08, 1.08),
        tuple(
            beinf_curve(*p)
            for p in ((0.5, 0.2, 0.2, 0.2), (0.3, 0.2, 0.8, 0.2), (0.7, 0.2, 0.2, 0.8))
        ),
    ),
    MixedChart(
        "zaga_component_mean_cv",
        "Zero-adjusted gamma (component mean/CV)",
        (-0.4, 40),
        tuple(zaga_curve(*p) for p in ((2, 0.5, 0.15), (4, 0.5, 0.35), (6, 0.7, 0.2))),
    ),
    MixedChart(
        "zaga_total_mean_cv",
        "Zero-adjusted gamma (total mean/CV)",
        (-0.4, 40),
        tuple(
            zaga_curve(*p, total=True)
            for p in ((2, 0.5, 0.15), (4, 0.5, 0.35), (6, 0.7, 0.2))
        ),
    ),
)


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=TITLE_PADDING)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=LABEL_FONT_SIZE)
    ax.grid(
        which="major",
        axis="both",
        color=GRID_COLOR,
        linestyle=GRID_LINE_STYLE,
        linewidth=GRID_LINE_WIDTH,
        alpha=GRID_ALPHA,
    )
    ax.spines[list(HIDDEN_SPINES)].set_visible(False)


def draw_continuous(chart: ContinuousChart, output_dir: Path, dpi: int) -> None:
    x = np.linspace(*chart.xlim, CURVE_SAMPLES)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout=FIGURE_LAYOUT)
    for color, curve in zip(COLORS, chart.curves, strict=False):
        ax.plot(x, curve.pdf(x), color=color, lw=CURVE_LINE_WIDTH, label=curve.label)
    style_axis(ax, chart.title, CONTINUOUS_Y_AXIS_LABEL)
    ax.set_xlim(*chart.xlim)
    ax.legend(**LEGEND_STYLE)
    chart_dir = output_dir / chart.slug
    chart_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_dir / f"{chart.slug}.png", dpi=dpi, facecolor=FIGURE_FACE_COLOR)
    fig.savefig(chart_dir / f"{chart.slug}.svg", facecolor=FIGURE_FACE_COLOR)
    plt.close(fig)


def draw_discrete(chart: DiscreteChart, output_dir: Path, dpi: int) -> None:
    k = np.arange(chart.xmax + 1)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout=FIGURE_LAYOUT)
    for color, (label, pmf) in zip(COLORS, chart.curves, strict=False):
        ax.vlines(k, 0, pmf(k), color=color, alpha=STEM_ALPHA, lw=STEM_LINE_WIDTH)
        ax.plot(
            k, pmf(k), MARKER_STYLE, color=color, ms=DISCRETE_MARKER_SIZE, label=label
        )
    style_axis(ax, chart.title, DISCRETE_Y_AXIS_LABEL)
    ax.set_xlim(-DISCRETE_X_MARGIN, chart.xmax + DISCRETE_X_MARGIN)
    ax.set_xticks(
        np.arange(0, chart.xmax + 1, max(1, chart.xmax // DISCRETE_TICK_COUNT))
    )
    ax.legend(**LEGEND_STYLE)
    chart_dir = output_dir / chart.slug
    chart_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_dir / f"{chart.slug}.png", dpi=dpi, facecolor=FIGURE_FACE_COLOR)
    fig.savefig(chart_dir / f"{chart.slug}.svg", facecolor=FIGURE_FACE_COLOR)
    plt.close(fig)


def draw_mixed(chart: MixedChart, output_dir: Path, dpi: int) -> None:
    """Построить непрерывную плотность и массы вероятности на концах."""
    x = np.linspace(*chart.xlim, CURVE_SAMPLES)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout=FIGURE_LAYOUT)
    for color, curve in zip(COLORS, chart.curves, strict=False):
        ax.plot(x, curve.pdf(x), color=color, lw=CURVE_LINE_WIDTH, label=curve.label)
        for endpoint, mass in zip(MASS_ENDPOINTS, curve.masses, strict=True):
            if mass:
                ax.vlines(
                    endpoint,
                    0,
                    mass,
                    color=color,
                    alpha=MASS_ALPHA,
                    lw=MASS_LINE_WIDTH,
                    linestyles=MASS_LINE_STYLE,
                )
                ax.plot(endpoint, mass, MARKER_STYLE, color=color, ms=MASS_MARKER_SIZE)
    style_axis(ax, chart.title, MIXED_Y_AXIS_LABEL)
    ax.set_xlim(*chart.xlim)
    ax.legend(**LEGEND_STYLE)
    chart_dir = output_dir / chart.slug
    chart_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_dir / f"{chart.slug}.png", dpi=dpi, facecolor=FIGURE_FACE_COLOR)
    fig.savefig(chart_dir / f"{chart.slug}.svg", facecolor=FIGURE_FACE_COLOR)
    plt.close(fig)


def draw_chart(slug: str, output_dir: Path, dpi: int) -> None:
    """Отрисовать один график; функция верхнего уровня для запуска в процессе-воркере."""
    for chart in CONTINUOUS:
        if chart.slug == slug:
            draw_continuous(chart, output_dir, dpi)
            return
    for chart in DISCRETE:
        if chart.slug == slug:
            draw_discrete(chart, output_dir, dpi)
            return
    for chart in MIXED:
        if chart.slug == slug:
            draw_mixed(chart, output_dir, dpi)
            return
    raise ValueError(f"Unknown chart: {slug}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("distribution_plots"))
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="resolution for PNG files (default: %(default)s)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="number of chart-rendering processes (default: %(default)s)",
    )
    parser.add_argument(
        "--silence",
        action="store_true",
        help="disable the progress bar",
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.dpi < 1:
        parser.error("--jobs and --dpi must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slugs = (
        [chart.slug for chart in CONTINUOUS]
        + [chart.slug for chart in DISCRETE]
        + [chart.slug for chart in MIXED]
    )
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for _ in tqdm(
            executor.map(
                draw_chart,
                slugs,
                [args.output_dir] * len(slugs),
                [args.dpi] * len(slugs),
            ),
            total=len(slugs),
            desc="Rendering charts",
            unit="chart",
            disable=args.silence,
        ):
            pass
    print(
        f"Created {len(slugs)} charts in {args.output_dir.resolve()} using {args.jobs} processes",
    )


if __name__ == "__main__":
    main()
