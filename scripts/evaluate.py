"""Evaluation script for Burgers PINN against exact spectral benchmark."""

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request
from typing import Any, Dict, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.stats import qmc
import torch
import yaml
from src.dataset import create_eval_grid, sample_extrapolation
from src.model import BurgersPINN
from src.physics import compute_pde_loss, compute_pde_residual


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_benchmark_data(benchmark_path: str, url: str) -> str:
    """Ensure benchmark .mat file exists locally, downloading if necessary."""
    if os.path.exists(benchmark_path):
        return benchmark_path

    # Check root workspace fallback
    if os.path.exists("burgers_shock.mat"):
        return "burgers_shock.mat"

    os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
    print(f"Downloading benchmark dataset from: {url}")
    urllib.request.urlretrieve(url, benchmark_path)
    print(f"Benchmark dataset downloaded to: {benchmark_path}")
    return benchmark_path


def load_benchmark(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load reference benchmark matrices from MATLAB .mat file."""
    data = scipy.io.loadmat(file_path)
    x_exact = data["x"].flatten()         # Shape: (256,) in [-1, 1]
    t_exact = data["t"].flatten()         # Shape: (100,) in [0, 1]
    u_exact = np.real(data["usol"])       # Shape: (256, 100)
    return x_exact, t_exact, u_exact


def plot_shock_parity(
    x_exact: np.ndarray,
    u_exact_t1: np.ndarray,
    u_pred_t1: np.ndarray,
    l2_error: float,
    output_path: str,
) -> None:
    """Plot exact vs predicted shock profile at final time t = 1.0."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_exact, u_exact_t1, "k--", linewidth=2.0, label="Exact (Chebyshev Spectral)")
    plt.plot(x_exact, u_pred_t1, "r-", linewidth=1.5, label=r"PINN Prediction ($\hat{u}$)")
    plt.title(f"Burgers' Shock Layer at $t = 1.0$ (Relative $L_2$ Error: {l2_error:.2e})")
    plt.xlabel("x")
    plt.ylabel("u(x, 1.0)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Shock parity plot saved to {output_path}")


def plot_solution_field(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    u_pred: np.ndarray,
    output_path: str,
) -> None:
    """Plot temporal evolution slices and 2D space-time heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Temporal Evolution Slices
    t_snapshots = [0.0, 0.25, 0.50, 0.75, 1.0]
    for t_val in t_snapshots:
        t_idx = int(t_val * (len(t_grid) - 1))
        ax1.plot(x_grid, u_pred[t_idx, :], label=f"t = {t_val:.2f}", linewidth=1.5)

    ax1.set_title("Burgers' Equation: Wave Profile Steepening")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x, t)")
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Space-Time Heatmap
    T, X = np.meshgrid(t_grid, x_grid)
    mesh = ax2.pcolormesh(T, X, u_pred.T, cmap="rainbow", shading="auto")
    ax2.set_title(r"Predicted Solution Field $\hat{u}(x, t)$")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x")
    fig.colorbar(mesh, ax=ax2, label="Velocity u")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Solution field plot saved to {output_path}")


def plot_error_field(
    x_exact: np.ndarray,
    t_exact: np.ndarray,
    u_exact: np.ndarray,
    u_pred_ref: np.ndarray,
    output_path: str,
) -> None:
    """Plot reference, predicted, and absolute error fields."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    T, X = np.meshgrid(t_exact, x_exact)
    abs_err = np.abs(u_exact - u_pred_ref)

    m1 = axes[0].pcolormesh(T, X, u_exact, cmap="rainbow", shading="auto")
    axes[0].set_title("Exact Solution $u(x, t)$")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    fig.colorbar(m1, ax=axes[0], label="u")

    m2 = axes[1].pcolormesh(T, X, u_pred_ref, cmap="rainbow", shading="auto")
    axes[1].set_title(r"PINN Prediction $\hat{u}(x, t)$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    fig.colorbar(m2, ax=axes[1], label="u")

    m3 = axes[2].pcolormesh(T, X, abs_err, cmap="viridis", shading="auto")
    axes[2].set_title(r"Absolute Pointwise Error $|u - \hat{u}|$")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    fig.colorbar(m3, ax=axes[2], label="Absolute Error")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Error field comparison saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Burgers PINN against exact benchmark.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint .pt")
    parser.add_argument("--benchmark-path", type=str, default=None, help="Path to burgers_shock.mat")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    checkpoint_file = cfg.get("paths", {}).get("checkpoint_file", "burgers_pinn.pt")
    checkpoint_path = args.checkpoint or os.path.join(checkpoint_dir, checkpoint_file)

    benchmark_path_cfg = cfg.get("evaluation", {}).get("benchmark_path", "data/burgers_shock.mat")
    benchmark_url = cfg.get("evaluation", {}).get("benchmark_url", "https://github.com/maziarraissi/PINNs/raw/master/appendix/Data/burgers_shock.mat")
    benchmark_path = args.benchmark_path or benchmark_path_cfg

    figures_dir = cfg.get("paths", {}).get("figures_dir", "figures")
    results_dir = cfg.get("paths", {}).get("results_dir", "results")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    nu = cfg.get("physics", {}).get("nu", 0.01 / np.pi)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_path}' not found. Please train first via `make train` or specify --checkpoint."
        )

    # 1. Load Model
    model_cfg = cfg.get("model", {})
    model = BurgersPINN(
        input_dim=model_cfg.get("input_dim", 2),
        output_dim=model_cfg.get("output_dim", 1),
        hidden_dim=model_cfg.get("hidden_dim", 50),
        hidden_layers=model_cfg.get("hidden_layers", 4),
        activation=model_cfg.get("activation", "tanh"),
    )
    ckpt_data = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from: {checkpoint_path}")

    # 2. Load Benchmark Ground Truth
    data_file = ensure_benchmark_data(benchmark_path, benchmark_url)
    x_exact, t_exact, u_exact = load_benchmark(data_file)
    print(f"Successfully loaded ground truth. Matrix shape: {u_exact.shape}")

    # 3. Model Inference on Exact Coordinates
    X_ref, T_ref = np.meshgrid(x_exact, t_exact)
    x_ref_flat = torch.tensor(X_ref.flatten()[:, None], dtype=torch.float32)
    t_ref_flat = torch.tensor(T_ref.flatten()[:, None], dtype=torch.float32)

    with torch.no_grad():
        u_pred_ref = model(x_ref_flat, t_ref_flat).numpy().reshape(len(t_exact), len(x_exact)).T

    # 4. Error Metrics
    abs_error = np.abs(u_exact - u_pred_ref)
    l2_error = float(np.linalg.norm(u_exact - u_pred_ref, 2) / np.linalg.norm(u_exact, 2))
    linf_error = float(np.max(abs_error))
    mae_error = float(np.mean(abs_error))

    # 5. In-Domain & Extrapolation PDE Residuals
    # In-domain regular grid residual
    x_grid = np.linspace(-1.0, 1.0, 256)
    t_grid = np.linspace(0.0, 1.0, 100)
    X_reg, T_reg = np.meshgrid(x_grid, t_grid)
    x_in_domain = torch.tensor(X_reg.flatten()[:, None], dtype=torch.float32, requires_grad=True)
    t_in_domain = torch.tensor(T_reg.flatten()[:, None], dtype=torch.float32, requires_grad=True)
    in_domain_res = compute_pde_residual(model, x_in_domain, t_in_domain, nu)
    in_domain_pde = float(torch.mean(in_domain_res ** 2).item())

    # Extrapolation points t in [1.0, 1.3]
    x_ext, t_ext = sample_extrapolation(n_ext=1000, x_bounds=(-1.0, 1.0), t_ext_bounds=(1.0, 1.3), seed=42)
    ext_res = compute_pde_residual(model, x_ext, t_ext, nu)
    ext_pde = float(torch.mean(ext_res ** 2).item())

    # 6. Generate Figures
    parity_path = os.path.join(figures_dir, "parity_t1.png")
    plot_shock_parity(x_exact, u_exact[:, -1], u_pred_ref[:, -1], l2_error, parity_path)

    # Dense regular grid for wave steepening slices
    with torch.no_grad():
        u_pred_dense = model(x_in_domain, t_in_domain).numpy().reshape(len(t_grid), len(x_grid))
    solution_field_path = os.path.join(figures_dir, "solution_field.png")
    plot_solution_field(x_grid, t_grid, u_pred_dense, solution_field_path)

    error_field_path = os.path.join(figures_dir, "error_field.png")
    plot_error_field(x_exact, t_exact, u_exact, u_pred_ref, error_field_path)

    # 7. Print and Save Summary
    metrics = {
        "relative_l2_error": l2_error,
        "relative_l2_error_percent": l2_error * 100.0,
        "max_absolute_error_linf": linf_error,
        "mean_absolute_error": mae_error,
        "in_domain_pde_residual": in_domain_pde,
        "extrapolation_pde_residual": ext_pde,
    }

    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("==================================================")
    print("           Evaluation Benchmark Results           ")
    print("==================================================")
    print(f"Overall Relative L2 Error:     {l2_error:.4e} ({l2_error * 100:.2f}%)")
    print(f"Maximum L_inf Error:           {linf_error:.4e}")
    print(f"Mean Absolute Error (MAE):     {mae_error:.4e}")
    print(f"In-Domain PDE Residual:        {in_domain_pde:.4e}")
    print(f"Extrapolation PDE Residual:    {ext_pde:.4e}")
    print(f"Metrics saved to:              {metrics_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
