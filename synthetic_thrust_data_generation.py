#!/usr/bin/env python3
"""
Synthetic thrust-fault seismic data generator with planar and Gaussian
background deformation.

This script corresponds to the step-by-step notebook:

    synthetic_thrust_fault_step_by_step_with_planar_gaussian.ipynb

Workflow
--------
1. Generate a flat 2D reflectivity model by laterally extending a random
   1D reflectivity sequence.

2. Optionally apply planar background deformation:
       s_planar(x) = c*x + e

3. Optionally apply Gaussian-shaped background deformation:
       s_gauss(x) = a0 + scale * sum_i b_i exp(-(x-x0_i)^2/(2 sigma_i^2))

4. Generate listric thrust-fault geometry:
       z = z0 + a(x-x0) + b[1-exp(-c(x-x0))]
   with:
       a in [0.27, 1.70]
       b in [10, 50] pixels
       c in [0.005, 0.02] pixel^-1
       a + b*c <= 1.73

5. Apply continuous thrust-fault displacement:
       u(x,z) = -d0/2 * exp(-F^2/(2 sigma^2)) * tanh(F/(2 lambda)) * t(x)

6. Generate synthetic seismic sections using either:
       - 1D vertical Ricker-wavelet convolution, or
       - 2D PSF convolution

7. Add Gaussian noise with std sampled from [0.2, 0.5].

Default output follows the manuscript-scale setting:
    12,000 training samples + 2,000 validation samples.

Array convention
----------------
The physical simulator uses ``(z, x)`` arrays internally because vertical
convolution and coordinate mapping are naturally expressed in that order.  The
public return value of ``generate_one_sample`` and every saved seismic/label
array use the repository-wide ``(x, z)`` convention.  ``height`` therefore
means the number of z samples and ``width`` means the number of x samples; a
saved array has shape ``(width, height)``.

Run the commands below from the repository root.

Quick demo
----------
python synthetic_thrust_data_generation.py \
    --out outputs/datasets/synthetic_demo \
    --num-train 8 \
    --num-val 2 \
    --preview 4

Full generation
---------------
python synthetic_thrust_data_generation.py \
    --out outputs/datasets/synthetic \
    --num-train 12000 \
    --num-val 2000
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage, signal

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


DATASET_SCHEMA_VERSION = "synthetic-thrust-dataset/2.0"
SAMPLE_SCHEMA_VERSION = "synthetic-thrust-sample/2.0"
OUTPUT_AXIS_ORDER = ("x", "z")


@dataclass
class SyntheticConfig:
    # Internal simulator shape is (height=z, width=x). Saved arrays are
    # transposed to the public (x, z) convention.
    height: int = 256
    width: int = 256

    # Dataset size.
    num_train: int = 12000
    num_val: int = 2000

    # Number of thrust faults per synthetic image.
    min_faults: int = 1
    max_faults: int = 3

    # Listric thrust-fault geometry.
    a_min: float = 0.27
    a_max: float = 1.70
    b_min: float = 10.0
    b_max: float = 50.0
    c_min: float = 0.005
    c_max: float = 0.02
    max_local_slope: float = 1.73

    # Continuous thrust displacement field.
    d0_min: float = 10.0
    d0_max: float = 90.0
    sigma_min: float = 10.0
    sigma_max: float = 50.0
    lambda_min: float = 1.0
    lambda_max: float = 5.0

    # Label width around the modeled fault core.
    label_half_width_min: float = 1.5
    label_half_width_max: float = 3.5

    # Planar background deformation.
    use_planar_background: bool = True
    planar_c_min: float = -0.08
    planar_c_max: float = 0.08
    planar_e_min: float = -8.0
    planar_e_max: float = 8.0

    # Gaussian-shaped background deformation.
    use_gaussian_background: bool = True
    min_gaussians: int = 1
    max_gaussians: int = 3
    gaussian_amp_min: float = -10.0
    gaussian_amp_max: float = 10.0
    gaussian_sigma_min: float = 20.0
    gaussian_sigma_max: float = 70.0
    gaussian_base_min: float = -5.0
    gaussian_base_max: float = 5.0
    gaussian_scale: float = 1.5

    # Seismic simulation.
    dt: float = 0.004
    ricker_freq_min: float = 15.0
    ricker_freq_max: float = 45.0
    wavelet_length: float = 0.128
    psf_sigma_x_min: float = 1.0
    psf_sigma_x_max: float = 3.5
    psf_sigma_z_min: float = 1.0
    psf_sigma_z_max: float = 3.5
    probability_2d_psf: float = 0.5

    # Additive Gaussian noise after normalization.
    noise_std_min: float = 0.2
    noise_std_max: float = 0.5

    # Random layered reflectivity model.
    min_layers: int = 12
    max_layers: int = 28

    # Augmentation preserving reverse/thrust style but changing dip direction.
    probability_horizontal_flip: float = 0.5

    seed: int = 2026


def normalize_section(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a section to zero mean and unit standard deviation."""
    x = x.astype(np.float32, copy=False)
    return (x - float(x.mean())) / (float(x.std()) + eps)


def ricker_wavelet(freq: float, dt: float, length: float) -> np.ndarray:
    """Return a zero-phase Ricker wavelet."""
    nt = int(round(length / dt))
    if nt % 2 == 0:
        nt += 1
    t = np.arange(-(nt // 2), nt // 2 + 1, dtype=np.float32) * dt
    pf2 = (math.pi * freq) ** 2
    w = (1.0 - 2.0 * pf2 * t**2) * np.exp(-pf2 * t**2)
    w /= np.max(np.abs(w)) + 1e-8
    return w.astype(np.float32)


def make_flat_reflectivity(cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a flat 2D reflectivity model by laterally extending a random
    1D reflectivity sequence.
    """
    h, w = cfg.height, cfg.width

    n_layers = int(rng.integers(cfg.min_layers, cfg.max_layers + 1))
    boundaries = np.sort(rng.choice(np.arange(5, h - 5), size=n_layers, replace=False))

    impedance = np.empty(h, dtype=np.float32)
    start = 0
    current = float(rng.uniform(1.0, 3.0))

    for b in list(boundaries) + [h]:
        current = max(0.2, current + float(rng.normal(0.0, 0.35)))
        impedance[start:b] = current
        start = int(b)

    impedance = ndimage.gaussian_filter1d(impedance, sigma=float(rng.uniform(0.6, 1.4)))

    refl_1d = np.zeros(h, dtype=np.float32)
    refl_1d[1:] = (impedance[1:] - impedance[:-1]) / (impedance[1:] + impedance[:-1] + 1e-8)

    weak = rng.normal(0.0, 0.015, size=h).astype(np.float32)
    weak = ndimage.gaussian_filter1d(weak, sigma=1.0)
    refl_1d += weak

    refl_2d = np.tile(refl_1d[:, None], (1, w))

    # Mild lateral amplitude variation keeps the model close to a 1D-extended
    # sequence while avoiding unrealistically identical traces.
    lateral = ndimage.gaussian_filter1d(rng.normal(1.0, 0.04, size=w), sigma=12.0)
    refl_2d = refl_2d * lateral[None, :]

    return refl_2d.astype(np.float32)


def vertical_warp_by_x_shift(cell: np.ndarray, shift_x: np.ndarray, order: int = 3) -> np.ndarray:
    """
    Resample each vertical trace using an x-dependent vertical shift.

    This is a CPU/SciPy equivalent of the sinc-interpolation idea used in the
    original CuPy `planar()` and `gaus()` functions.

    For each output location (z, x), the input is sampled at:
        z_input = z + shift_x[x]
    """
    h, w = cell.shape
    x_grid, z_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    coords_z = z_grid + shift_x[None, :].astype(np.float32)
    coords_x = x_grid

    warped = ndimage.map_coordinates(
        cell,
        [coords_z, coords_x],
        order=order,
        mode="nearest",
    )
    return warped.astype(np.float32)


def planar_background_deformation(
    cfg: SyntheticConfig,
    cell: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply planar background deformation:
        s(x) = c*x + e
    """
    x = np.arange(cfg.width, dtype=np.float32)

    c = float(rng.uniform(cfg.planar_c_min, cfg.planar_c_max))
    e = float(rng.uniform(cfg.planar_e_min, cfg.planar_e_max))

    shift_x = c * x + e
    warped = vertical_warp_by_x_shift(cell, shift_x, order=3)

    params = {
        "type": "planar",
        "c": c,
        "e": e,
        "shift_min": float(shift_x.min()),
        "shift_max": float(shift_x.max()),
    }
    return warped.astype(np.float32), params


def gaussian_background_deformation(
    cfg: SyntheticConfig,
    cell: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply Gaussian-shaped background deformation:
        s(x) = base + scale * sum_i amp_i exp(-(x-x0_i)^2/(2 sigma_i^2))
    """
    x = np.arange(cfg.width, dtype=np.float32)

    n_gaussians = int(rng.integers(cfg.min_gaussians, cfg.max_gaussians + 1))
    centers = rng.uniform(0.05 * cfg.width, 0.95 * cfg.width, size=n_gaussians)
    amplitudes = rng.uniform(cfg.gaussian_amp_min, cfg.gaussian_amp_max, size=n_gaussians)
    sigmas = rng.uniform(cfg.gaussian_sigma_min, cfg.gaussian_sigma_max, size=n_gaussians)
    base = float(rng.uniform(cfg.gaussian_base_min, cfg.gaussian_base_max))

    gauss_1d = np.zeros(cfg.width, dtype=np.float32)
    for x0, amp, sigma in zip(centers, amplitudes, sigmas):
        gauss_1d += amp * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))

    shift_x = base + cfg.gaussian_scale * gauss_1d
    warped = vertical_warp_by_x_shift(cell, shift_x, order=3)

    params = {
        "type": "gaussian_background",
        "base": base,
        "scale": cfg.gaussian_scale,
        "centers": centers.tolist(),
        "amplitudes": amplitudes.tolist(),
        "sigmas": sigmas.tolist(),
        "shift_min": float(shift_x.min()),
        "shift_max": float(shift_x.max()),
    }
    return warped.astype(np.float32), params


def apply_background_deformation(
    cfg: SyntheticConfig,
    refl: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply optional planar and Gaussian background deformation before thrust faulting.
    """
    background = refl.copy()
    metadata = {"steps": []}

    if cfg.use_planar_background:
        background, params = planar_background_deformation(cfg, background, rng)
        metadata["steps"].append(params)

    if cfg.use_gaussian_background:
        background, params = gaussian_background_deformation(cfg, background, rng)
        metadata["steps"].append(params)

    return background.astype(np.float32), metadata


def sample_listric_fault(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    max_tries: int = 500,
) -> Dict[str, float]:
    """
    Sample listric thrust-fault parameters satisfying a + b*c <= 1.73 and keeping
    a sufficiently long visible fault segment inside the image.
    """
    h, w = cfg.height, cfg.width

    for _ in range(max_tries):
        a = float(rng.uniform(cfg.a_min, cfg.a_max))
        b = float(rng.uniform(cfg.b_min, cfg.b_max))
        c = float(rng.uniform(cfg.c_min, cfg.c_max))

        if a + b * c > cfg.max_local_slope:
            continue

        x0 = float(rng.uniform(0.02 * w, 0.45 * w))
        z0 = float(rng.uniform(0.02 * h, 0.65 * h))

        x = np.arange(w, dtype=np.float32)
        f = z0 + a * (x - x0) + b * (1.0 - np.exp(-c * (x - x0)))
        visible = np.logical_and(f >= 0, f < h)

        if int(visible.sum()) >= max(50, int(0.22 * w)):
            return {"x0": x0, "z0": z0, "a": a, "b": b, "c": c}

    raise RuntimeError("Could not sample a valid listric thrust fault.")


def listric_curve_and_slope(
    x: np.ndarray,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the listric fault curve and local slope."""
    x0, z0 = params["x0"], params["z0"]
    a, b, c = params["a"], params["b"], params["c"]
    q = x - x0
    f = z0 + a * q + b * (1.0 - np.exp(-c * q))
    fp = a + b * c * np.exp(-c * q)
    return f.astype(np.float32), fp.astype(np.float32)


def displacement_field_for_fault(
    cfg: SyntheticConfig,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Construct the continuous thrust-fault displacement field.

    Returns:
        ux: horizontal displacement
        uz: vertical displacement
        label: finite-width fault-zone label
        full_params: sampled fault and displacement parameters
    """
    h, w = cfg.height, cfg.width
    x = np.arange(w, dtype=np.float32)
    _, z_grid = np.meshgrid(x, np.arange(h, dtype=np.float32))

    f, fp = listric_curve_and_slope(x, params)
    F = z_grid - f[None, :]

    denom = np.sqrt(1.0 + fp**2)
    tx = 1.0 / denom
    tz = fp / denom

    d0 = float(rng.uniform(cfg.d0_min, cfg.d0_max))
    sigma = float(rng.uniform(cfg.sigma_min, cfg.sigma_max))
    lam = float(rng.uniform(cfg.lambda_min, cfg.lambda_max))

    amplitude = -0.5 * d0 * np.exp(-(F**2) / (2.0 * sigma**2)) * np.tanh(F / (2.0 * lam))
    ux = amplitude * tx[None, :]
    uz = amplitude * tz[None, :]

    # A narrow finite-width label is more suitable for segmentation training than
    # a one-pixel fault curve.
    normal_distance = np.abs(F) / denom[None, :]
    label_half_width = float(rng.uniform(cfg.label_half_width_min, cfg.label_half_width_max))
    valid_curve = np.logical_and(f >= 0, f < h)
    label = np.logical_and(normal_distance <= label_half_width, valid_curve[None, :])

    full_params = dict(params)
    full_params.update({
        "d0": d0,
        "sigma": sigma,
        "lambda": lam,
        "label_half_width": label_half_width,
    })
    return ux.astype(np.float32), uz.astype(np.float32), label.astype(np.uint8), full_params


def warp_reflectivity(
    refl: np.ndarray,
    ux: np.ndarray,
    uz: np.ndarray,
) -> np.ndarray:
    """
    Warp reflectivity using backward mapping.
    """
    h, w = refl.shape
    x_grid, z_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    coords_z = z_grid - uz
    coords_x = x_grid - ux

    warped = ndimage.map_coordinates(
        refl,
        [coords_z, coords_x],
        order=1,
        mode="nearest",
    )
    return warped.astype(np.float32)


def apply_thrust_faults(
    cfg: SyntheticConfig,
    refl: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    Apply one or more listric thrust-fault displacement fields.
    """
    num_faults = int(rng.integers(cfg.min_faults, cfg.max_faults + 1))

    deformed = refl.copy()
    labels = np.zeros_like(refl, dtype=np.uint8)
    params_all: List[Dict[str, float]] = []

    for _ in range(num_faults):
        params = sample_listric_fault(cfg, rng)
        ux, uz, label, full_params = displacement_field_for_fault(cfg, params, rng)
        deformed = warp_reflectivity(deformed, ux, uz)
        labels = np.maximum(labels, label)
        params_all.append(full_params)

    if rng.random() < cfg.probability_horizontal_flip:
        # Internal arrays are (z, x), so physical x is axis 1 here. The public
        # output is transposed to (x, z) later in generate_one_sample().
        deformed = np.flip(deformed, axis=1).copy()
        labels = np.flip(labels, axis=1).copy()
        for p in params_all:
            p["horizontally_flipped"] = True
    else:
        for p in params_all:
            p["horizontally_flipped"] = False

    return deformed.astype(np.float32), labels.astype(np.uint8), params_all


def convolve_1d_vertical(
    cfg: SyntheticConfig,
    refl: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Convolve each vertical trace with a 1D Ricker wavelet."""
    freq = float(rng.uniform(cfg.ricker_freq_min, cfg.ricker_freq_max))
    wavelet = ricker_wavelet(freq=freq, dt=cfg.dt, length=cfg.wavelet_length)
    seismic = ndimage.convolve1d(refl, weights=wavelet, axis=0, mode="nearest")
    return seismic.astype(np.float32), {"operator": "1d_ricker", "frequency_hz": freq}


def make_2d_psf(
    cfg: SyntheticConfig,
    freq: float,
    sigma_x: float,
    sigma_z: float,
) -> np.ndarray:
    """
    Create a simple 2D PSF: a vertical Ricker wavelet modulated by an anisotropic
    Gaussian in x and z.
    """
    wavelet = ricker_wavelet(freq=freq, dt=cfg.dt, length=cfg.wavelet_length)
    kz = len(wavelet)
    kx = int(max(9, math.ceil(8 * sigma_x)))
    if kx % 2 == 0:
        kx += 1

    z = np.arange(-(kz // 2), kz // 2 + 1, dtype=np.float32)
    x = np.arange(-(kx // 2), kx // 2 + 1, dtype=np.float32)
    x_grid, z_grid = np.meshgrid(x, z)

    gx = np.exp(-(x_grid**2) / (2.0 * sigma_x**2))
    gz = np.exp(-(z_grid**2) / (2.0 * sigma_z**2))
    psf = wavelet[:, None] * gx * gz
    psf -= psf.mean()
    psf /= np.sum(np.abs(psf)) + 1e-8
    return psf.astype(np.float32)


def convolve_2d_psf(
    cfg: SyntheticConfig,
    refl: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Convolve reflectivity with a 2D point-spread function."""
    freq = float(rng.uniform(cfg.ricker_freq_min, cfg.ricker_freq_max))
    sigma_x = float(rng.uniform(cfg.psf_sigma_x_min, cfg.psf_sigma_x_max))
    sigma_z = float(rng.uniform(cfg.psf_sigma_z_min, cfg.psf_sigma_z_max))

    psf = make_2d_psf(cfg, freq=freq, sigma_x=sigma_x, sigma_z=sigma_z)
    seismic = signal.fftconvolve(refl, psf, mode="same")
    return seismic.astype(np.float32), {
        "operator": "2d_psf",
        "frequency_hz": freq,
        "psf_sigma_x": sigma_x,
        "psf_sigma_z": sigma_z,
    }


def add_gaussian_noise(
    cfg: SyntheticConfig,
    seismic: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    Normalize, add Gaussian noise, normalize again, and clip to [-1, 1].
    """
    seismic = normalize_section(seismic)
    noise_std = float(rng.uniform(cfg.noise_std_min, cfg.noise_std_max))
    noisy = seismic + rng.normal(0.0, noise_std, size=seismic.shape).astype(np.float32)
    noisy = normalize_section(noisy)
    noisy = np.clip(noisy, -3.0, 3.0) / 3.0
    return noisy.astype(np.float32), noise_std


def simulate_seismic(
    cfg: SyntheticConfig,
    deformed_refl: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate a synthetic seismic image using either 1D vertical convolution or
    2D PSF convolution.
    """
    if rng.random() < cfg.probability_2d_psf:
        seismic, sim_params = convolve_2d_psf(cfg, deformed_refl, rng)
    else:
        seismic, sim_params = convolve_1d_vertical(cfg, deformed_refl, rng)

    seismic, noise_std = add_gaussian_noise(cfg, seismic, rng)
    sim_params["noise_std"] = noise_std
    return seismic.astype(np.float32), sim_params


def generate_one_sample(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate one seismic-label pair and metadata.

    Complete workflow:
        flat reflectivity
        -> planar/Gaussian background deformation
        -> thrust-fault displacement
        -> seismic convolution
        -> Gaussian noise
    """
    flat_zx = make_flat_reflectivity(cfg, rng)
    background_zx, background_params = apply_background_deformation(cfg, flat_zx, rng)
    deformed_zx, label_zx, fault_params = apply_thrust_faults(cfg, background_zx, rng)
    seismic_zx, sim_params = simulate_seismic(cfg, deformed_zx, rng)

    expected_internal_shape = (cfg.height, cfg.width)
    if seismic_zx.shape != expected_internal_shape:
        raise RuntimeError(
            "Internal seismic shape mismatch: "
            f"received {seismic_zx.shape}, expected {expected_internal_shape} in (z, x) order."
        )
    if label_zx.shape != expected_internal_shape:
        raise RuntimeError(
            "Internal label shape mismatch: "
            f"received {label_zx.shape}, expected {expected_internal_shape} in (z, x) order."
        )

    # This is the single public axis-conversion boundary. Keep all physical
    # modeling above in (z, x), but return and save contiguous (x, z) arrays so
    # synthetic and real training patches share the same convention.
    seismic_xz = np.ascontiguousarray(seismic_zx.T, dtype=np.float32)
    label_xz = np.ascontiguousarray(label_zx.T, dtype=np.uint8)

    metadata = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "array_axis_order": list(OUTPUT_AXIS_ORDER),
        "axis_sizes": {"x": cfg.width, "z": cfg.height},
        "background_deformation": background_params,
        "faults": fault_params,
        "seismic_simulation": sim_params,
        "image_shape": [cfg.width, cfg.height],
    }
    return seismic_xz, label_xz, metadata


def save_preview_png(
    out_png: Path,
    seismic: np.ndarray,
    label: np.ndarray,
) -> None:
    """Save a quick preview for seismic and label arrays in (x, z) order."""
    import matplotlib.pyplot as plt

    if seismic.shape != label.shape:
        raise ValueError(
            f"Preview seismic/label shape mismatch: {seismic.shape} vs {label.shape}."
        )
    nx, nz = seismic.shape

    plt.figure(figsize=(5, 5))
    plt.imshow(seismic.T, cmap="gray", aspect="auto", extent=[0, nx, nz, 0])
    xs, zs = np.where(label > 0)
    if len(xs) > 0:
        plt.scatter(xs, zs, s=1)
    plt.xlim(0, nx)
    plt.ylim(nz, 0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=150)
    plt.close()


def write_split(
    cfg: SyntheticConfig,
    out_dir: Path,
    split: str,
    n: int,
    rng: np.random.Generator,
    preview: int = 0,
) -> None:
    """Generate and save one dataset split."""
    image_dir = out_dir / split / "seismic"
    label_dir = out_dir / split / "labels"
    meta_dir = out_dir / split / "metadata"
    preview_dir = out_dir / split / "preview"

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if preview > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n), desc=f"Generating {split}"):
        seismic, label, metadata = generate_one_sample(cfg, rng)

        expected_output_shape = (cfg.width, cfg.height)
        if seismic.shape != expected_output_shape or label.shape != expected_output_shape:
            raise RuntimeError(
                "Synthetic output contract violation: seismic and label must both "
                f"have shape {expected_output_shape} in (x, z) order; received "
                f"{seismic.shape} and {label.shape}."
            )
        if metadata.get("array_axis_order") != list(OUTPUT_AXIS_ORDER):
            raise RuntimeError("Synthetic metadata does not declare (x, z) axis order.")

        stem = f"{split}_{i:06d}"
        np.save(image_dir / f"{stem}.npy", seismic.astype(np.float32))
        np.save(label_dir / f"{stem}.npy", label.astype(np.uint8))

        with open(meta_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if i < preview:
            save_preview_png(preview_dir / f"{stem}.png", seismic, label)


def assert_empty_output_directory(out_dir: Path) -> None:
    """Refuse to mix newly generated (x, z) samples with an older dataset."""
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {out_dir}")

    existing_entries = sorted(
        (
            path
            for path in out_dir.iterdir()
            if not (
                path.name == ".gitignore"
                and path.is_file()
                and not path.is_symlink()
            )
        ),
        key=lambda path: path.name,
    )
    if existing_entries:
        preview = ", ".join(path.name for path in existing_entries[:5])
        suffix = " ..." if len(existing_entries) > 5 else ""
        raise FileExistsError(
            f"Output directory {out_dir} is not empty ({preview}{suffix}). "
            "Generate the (x, z) dataset in a new or empty directory so legacy "
            "(z, x) samples cannot be mixed with current samples. Only a regular "
            "repository .gitignore file is allowed."
        )


def run_self_test() -> None:
    """Verify the public (x, z) contract with a deliberately non-square sample."""
    cfg = SyntheticConfig(
        height=96,
        width=128,
        num_train=1,
        num_val=0,
        min_faults=1,
        max_faults=1,
        seed=2026,
    )
    seismic, label, metadata = generate_one_sample(cfg, np.random.default_rng(cfg.seed))
    expected_shape = (cfg.width, cfg.height)

    if seismic.shape != expected_shape or label.shape != expected_shape:
        raise AssertionError(
            f"Expected non-square (x, z) shape {expected_shape}, received "
            f"{seismic.shape} and {label.shape}."
        )
    if not seismic.flags.c_contiguous or not label.flags.c_contiguous:
        raise AssertionError("Public seismic and label arrays must be C-contiguous.")
    if seismic.dtype != np.float32 or label.dtype != np.uint8:
        raise AssertionError(f"Unexpected dtypes: {seismic.dtype}, {label.dtype}.")
    if not np.isfinite(seismic).all() or not np.isfinite(label).all():
        raise AssertionError("Synthetic self-test produced non-finite values.")
    if not np.any(label):
        raise AssertionError("Synthetic self-test produced an empty fault label.")
    if metadata.get("array_axis_order") != list(OUTPUT_AXIS_ORDER):
        raise AssertionError("Synthetic metadata axis order is not (x, z).")
    if metadata.get("image_shape") != list(expected_shape):
        raise AssertionError("Synthetic metadata image shape does not match the arrays.")

    print(f"Synthetic axis self-test passed: shape={expected_shape}, order=(x, z)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic thrust-fault seismic data with planar/Gaussian background deformation."
    )

    parser.add_argument(
        "--out",
        type=str,
        default="outputs/datasets/synthetic",
        help="Output dataset directory (default: outputs/datasets/synthetic).",
    )
    parser.add_argument("--num-train", type=int, default=12000)
    parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Number of z samples; saved arrays use this as axis 1 (default: 256).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Number of x samples; saved arrays use this as axis 0 (default: 256).",
    )
    parser.add_argument("--min-faults", type=int, default=1)
    parser.add_argument("--max-faults", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--preview", type=int, default=0, help="Number of preview PNGs per split.")

    parser.add_argument(
        "--no-planar-background",
        action="store_true",
        help="Disable planar background deformation.",
    )
    parser.add_argument(
        "--no-gaussian-background",
        action="store_true",
        help="Disable Gaussian background deformation.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Generate one non-square in-memory sample and verify the (x, z) contract.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.self_test:
        run_self_test()
        return

    cfg = SyntheticConfig(
        height=args.height,
        width=args.width,
        num_train=args.num_train,
        num_val=args.num_val,
        min_faults=args.min_faults,
        max_faults=args.max_faults,
        use_planar_background=not args.no_planar_background,
        use_gaussian_background=not args.no_gaussian_background,
        seed=args.seed,
    )

    out_dir = Path(args.out)
    assert_empty_output_directory(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = asdict(cfg)
    dataset_config.update(
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "array_axis_order": list(OUTPUT_AXIS_ORDER),
            "array_shape": [cfg.width, cfg.height],
            "axis_sizes": {"x": cfg.width, "z": cfg.height},
        }
    )
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataset_config, f, indent=2)

    rng = np.random.default_rng(cfg.seed)
    write_split(cfg, out_dir, "train", cfg.num_train, rng, preview=args.preview)
    write_split(cfg, out_dir, "val", cfg.num_val, rng, preview=args.preview)

    print(f"Done. Dataset written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
