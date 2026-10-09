"""Automated ablation study sweeps for Burgers PINN."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scripts.evaluate import ensure_benchmark_data, load_benchmark
from src.dataset import create_burgers_dataset, sample_extrapolation
from src.model import BurgersPINN
from src.physics import compute_loss, compute_pde_residual
from src.trainer import PINNTrainer


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model_on_benchmark(
    model: torch.nn.Module,
    x_exact: np.ndarray,
    t_exact: np.ndarray,
    u_exact: np.ndarray,
    nu: float,
) -> Dict[str, float]:
    """Evaluate model for relative L2 error, in-domain residual, and extrapolation residual."""
    model.eval()
    X_ref, T_ref = np.meshgrid(x_exact, t_exact)
    x_flat = torch.tensor(X_ref.flatten()[:, None], dtype=torch.float32)
    t_flat = torch.tensor(T_ref.flatten()[:, None], dtype=torch.float32)

    with torch.no_grad():
        u_pred = model(x_flat, t_flat).numpy().reshape(len(t_exact), len(x_exact)).T

    l2_error = float(np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2))

    # In-domain PDE residual
    x_in = torch.tensor(X_ref.flatten()[:, None], dtype=torch.float32, requires_grad=True)
    t_in = torch.tensor(T_ref.flatten()[:, None], dtype=torch.float32, requires_grad=True)
    res_in = compute_pde_residual(model, x_in, t_in, nu)
    in_domain_pde = float(torch.mean(res_in ** 2).item())

    # Extrapolation PDE residual
    x_ext, t_ext = sample_extrapolation(n_ext=1000, x_bounds=(-1.0, 1.0), t_ext_bounds=(1.0, 1.3), seed=42)
    res_ext = compute_pde_residual(model, x_ext, t_ext, nu)
    ext_pde = float(torch.mean(res_ext ** 2).item())

    return {
        "relative_l2_error": l2_error,
        "relative_l2_percent": l2_error * 100.0,
        "in_domain_pde_residual": in_domain_pde,
        "extrapolation_pde_residual": ext_pde,
    }


# =====================================================================
# Sweep 1: Optimization Strategy
# =====================================================================
def run_optimizer_sweep(
    cfg: Dict[str, Any],
    benchmark_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    device: str = "cpu",
    fast: bool = False,
) -> List[Dict[str, Any]]:
    print("\n>>> Running Sweep 1: Optimization Strategy (Adam vs L-BFGS vs Staged)")
    x_exact, t_exact, u_exact = benchmark_data
    nu = cfg.get("physics", {}).get("nu", 0.01 / np.pi)
    seed = cfg.get("seed", 42)

    adam_epochs = 300 if fast else cfg.get("training", {}).get("adam_epochs", 2000)
    lbfgs_iter = 150 if fast else cfg.get("training", {}).get("lbfgs_max_iter", 1000)

    strategies = [
        {"name": "Adam Only", "stages": "adam"},
        {"name": "L-BFGS Only", "stages": "lbfgs"},
        {"name": "Staged Adam + L-BFGS", "stages": "staged"},
    ]

    results = []
    for strat in strategies:
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = create_burgers_dataset(
            n_f=cfg.get("collocation", {}).get("n_f", 2500),
            n_bc=cfg.get("collocation", {}).get("n_bc", 100),
            n_ic=cfg.get("collocation", {}).get("n_ic", 100),
            seed=seed,
        )
        model = BurgersPINN(hidden_dim=50, hidden_layers=4)
        trainer = PINNTrainer(
            model=model,
            dataset=dataset,
            nu=nu,
            w_ic=cfg.get("loss_weights", {}).get("w_ic", 50.0),
            w_bc=cfg.get("loss_weights", {}).get("w_bc", 10.0),
            w_pde=cfg.get("loss_weights", {}).get("w_pde", 1.0),
            device=device,
        )

        t0 = time.time()
        if strat["stages"] == "adam":
            trainer.train_adam(epochs=adam_epochs + (lbfgs_iter if not fast else 0), lr=1e-3, verbose=False)
        elif strat["stages"] == "lbfgs":
            trainer.train_lbfgs(max_iter=lbfgs_iter, verbose=False)
        else:
            trainer.train_staged(adam_epochs=adam_epochs, lbfgs_max_iter=lbfgs_iter, verbose=False)
        elapsed = time.time() - t0

        eval_res = evaluate_model_on_benchmark(model, x_exact, t_exact, u_exact, nu)
        loss_bd = trainer.compute_current_loss()

        record = {
            "strategy": strat["name"],
            "wall_clock_sec": elapsed,
            "total_loss": loss_bd.total.item(),
            "pde_loss": loss_bd.pde.item(),
            "relative_l2_percent": eval_res["relative_l2_percent"],
            "in_domain_pde": eval_res["in_domain_pde_residual"],
            "extrapolation_pde": eval_res["extrapolation_pde_residual"],
        }
        results.append(record)
        print(f"  {strat['name']:22s} | Rel L2: {record['relative_l2_percent']:6.2f}% | "
              f"PDE Res: {record['in_domain_pde']:.4e} | Time: {elapsed:.1f}s")

    return results


# =====================================================================
# Sweep 2: Loss Weighting
# =====================================================================
def run_loss_weight_sweep(
    cfg: Dict[str, Any],
    benchmark_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    device: str = "cpu",
    fast: bool = False,
) -> List[Dict[str, Any]]:
    print("\n>>> Running Sweep 2: Loss Weight Configurations (w_ic, w_bc, w_pde)")
    x_exact, t_exact, u_exact = benchmark_data
    nu = cfg.get("physics", {}).get("nu", 0.01 / np.pi)
    seed = cfg.get("seed", 42)

    adam_epochs = 300 if fast else cfg.get("training", {}).get("adam_epochs", 2000)
    lbfgs_iter = 100 if fast else cfg.get("training", {}).get("lbfgs_max_iter", 1000)

    weight_configs = [
        {"name": "Equal (1:1:1)", "w_ic": 1.0, "w_bc": 1.0, "w_pde": 1.0},
        {"name": "Physics-Heavy (1:1:10)", "w_ic": 1.0, "w_bc": 1.0, "w_pde": 10.0},
        {"name": "Moderate IC/BC (10:10:1)", "w_ic": 10.0, "w_bc": 10.0, "w_pde": 1.0},
        {"name": "Notebook Baseline (50:10:1)", "w_ic": 50.0, "w_bc": 10.0, "w_pde": 1.0},
        {"name": "IC-Heavy (100:10:1)", "w_ic": 100.0, "w_bc": 10.0, "w_pde": 1.0},
    ]

    results = []
    for wc in weight_configs:
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = create_burgers_dataset(
            n_f=cfg.get("collocation", {}).get("n_f", 2500),
            n_bc=cfg.get("collocation", {}).get("n_bc", 100),
            n_ic=cfg.get("collocation", {}).get("n_ic", 100),
            seed=seed,
        )
        model = BurgersPINN(hidden_dim=50, hidden_layers=4)
        trainer = PINNTrainer(
            model=model,
            dataset=dataset,
            nu=nu,
            w_ic=wc["w_ic"],
            w_bc=wc["w_bc"],
            w_pde=wc["w_pde"],
            device=device,
        )

        t0 = time.time()
        trainer.train_staged(adam_epochs=adam_epochs, lbfgs_max_iter=lbfgs_iter, verbose=False)
        elapsed = time.time() - t0

        eval_res = evaluate_model_on_benchmark(model, x_exact, t_exact, u_exact, nu)
        loss_bd = trainer.compute_current_loss()

        record = {
            "config": wc["name"],
            "w_ic": wc["w_ic"],
            "w_bc": wc["w_bc"],
            "w_pde": wc["w_pde"],
            "loss_ic": loss_bd.ic.item(),
            "loss_bc": loss_bd.bc.item(),
            "loss_pde": loss_bd.pde.item(),
            "relative_l2_percent": eval_res["relative_l2_percent"],
            "in_domain_pde": eval_res["in_domain_pde_residual"],
        }
        results.append(record)
        print(f"  {wc['name']:25s} | Rel L2: {record['relative_l2_percent']:6.2f}% | "
              f"IC: {record['loss_ic']:.3e} | BC: {record['loss_bc']:.3e} | PDE: {record['loss_pde']:.3e}")

    return results


# =====================================================================
# Sweep 3: Collocation Point Density
# =====================================================================
def run_collocation_sweep(
    cfg: Dict[str, Any],
    benchmark_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    device: str = "cpu",
    fast: bool = False,
) -> List[Dict[str, Any]]:
    print("\n>>> Running Sweep 3: Collocation Point Density (N_f)")
    x_exact, t_exact, u_exact = benchmark_data
    nu = cfg.get("physics", {}).get("nu", 0.01 / np.pi)
    seed = cfg.get("seed", 42)

    adam_epochs = 300 if fast else cfg.get("training", {}).get("adam_epochs", 2000)
    lbfgs_iter = 100 if fast else cfg.get("training", {}).get("lbfgs_max_iter", 1000)

    n_f_counts = [500, 1000, 2500, 5000]

    results = []
    for n_f in n_f_counts:
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = create_burgers_dataset(
            n_f=n_f,
            n_bc=cfg.get("collocation", {}).get("n_bc", 100),
            n_ic=cfg.get("collocation", {}).get("n_ic", 100),
            seed=seed,
        )
        model = BurgersPINN(hidden_dim=50, hidden_layers=4)
        trainer = PINNTrainer(
            model=model,
            dataset=dataset,
            nu=nu,
            w_ic=cfg.get("loss_weights", {}).get("w_ic", 50.0),
            w_bc=cfg.get("loss_weights", {}).get("w_bc", 10.0),
            w_pde=cfg.get("loss_weights", {}).get("w_pde", 1.0),
            device=device,
        )

        t0 = time.time()
        trainer.train_staged(adam_epochs=adam_epochs, lbfgs_max_iter=lbfgs_iter, verbose=False)
        elapsed = time.time() - t0

        eval_res = evaluate_model_on_benchmark(model, x_exact, t_exact, u_exact, nu)

        record = {
            "n_collocation": n_f,
            "wall_clock_sec": elapsed,
            "relative_l2_percent": eval_res["relative_l2_percent"],
            "in_domain_pde": eval_res["in_domain_pde_residual"],
            "extrapolation_pde": eval_res["extrapolation_pde_residual"],
        }
        results.append(record)
        print(f"  N_f = {n_f:5d} | Rel L2: {record['relative_l2_percent']:6.2f}% | "
              f"PDE Res: {record['in_domain_pde']:.4e} | Extrap: {record['extrapolation_pde']:.4e} | Time: {elapsed:.1f}s")

    return results


def plot_ablation_summaries(
    opt_results: List[Dict[str, Any]],
    weight_results: List[Dict[str, Any]],
    colloc_results: List[Dict[str, Any]],
    figures_dir: str,
) -> None:
    """Generate and save comparison charts for all completed sweeps."""
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Optimizer Strategy Plot
    if opt_results:
        plt.figure(figsize=(7, 4.5))
        names = [r["strategy"] for r in opt_results]
        errors = [r["relative_l2_percent"] for r in opt_results]
        bars = plt.bar(names, errors, color=["#4A90E2", "#F5A623", "#50E3C2"])
        plt.ylabel("Relative $L_2$ Error (%)")
        plt.title("Ablation 1: Optimization Strategy Comparison")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.2f}%", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "ablation_optimizer.png"), dpi=300)
        plt.close()

    # 2. Loss Weighting Plot
    if weight_results:
        plt.figure(figsize=(8, 4.5))
        names = [r["config"] for r in weight_results]
        errors = [r["relative_l2_percent"] for r in weight_results]
        bars = plt.bar(names, errors, color="#7B68EE")
        plt.ylabel("Relative $L_2$ Error (%)")
        plt.xticks(rotation=20, ha="right")
        plt.title(r"Ablation 2: Loss Weight Configurations ($w_{ic}, w_{bc}, w_{pde}$)")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.2f}%", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "ablation_weights.png"), dpi=300)
        plt.close()

    # 3. Collocation Density Plot
    if colloc_results:
        plt.figure(figsize=(7, 4.5))
        counts = [r["n_collocation"] for r in colloc_results]
        errors = [r["relative_l2_percent"] for r in colloc_results]
        plt.plot(counts, errors, "o-", color="#E94E77", linewidth=2.0, markersize=8)
        plt.xlabel("Collocation Points ($N_f$)")
        plt.ylabel("Relative $L_2$ Error (%)")
        plt.title("Ablation 3: Collocation Point Density Scaling")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "ablation_collocation.png"), dpi=300)
        plt.close()


def generate_markdown_report(
    opt_results: List[Dict[str, Any]],
    weight_results: List[Dict[str, Any]],
    colloc_results: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """Format Markdown tables of ablation findings."""
    lines = ["# Burgers PINN Ablation Study Results\n"]

    if opt_results:
        lines.append("## 1. Optimization Strategy Sweep\n")
        lines.append("| Strategy | Relative $L_2$ Error (%) | In-Domain PDE Residual | Wall Time (s) |")
        lines.append("|---|:---:|:---:|:---:|")
        for r in opt_results:
            lines.append(f"| {r['strategy']} | {r['relative_l2_percent']:.2f}% | {r['in_domain_pde']:.4e} | {r['wall_clock_sec']:.1f}s |")
        lines.append("")

    if weight_results:
        lines.append("## 2. Loss Weight Configurations Sweep\n")
        lines.append("| Weight Configuration ($w_{ic}:w_{bc}:w_{pde}$) | Relative $L_2$ Error (%) | PDE Loss | IC Loss | BC Loss |")
        lines.append("|---|:---:|:---:|:---:|:---:|")
        for r in weight_results:
            lines.append(f"| {r['config']} | {r['relative_l2_percent']:.2f}% | {r['loss_pde']:.4e} | {r['loss_ic']:.4e} | {r['loss_bc']:.4e} |")
        lines.append("")

    if colloc_results:
        lines.append("## 3. Collocation Point Density Sweep\n")
        lines.append("| Collocation Count ($N_f$) | Relative $L_2$ Error (%) | In-Domain PDE Residual | Extrapolation Residual ($t > 1$) | Wall Time (s) |")
        lines.append("|---|:---:|:---:|:---:|:---:|")
        for r in colloc_results:
            lines.append(f"| {r['n_collocation']} | {r['relative_l2_percent']:.2f}% | {r['in_domain_pde']:.4e} | {r['extrapolation_pde']:.4e} | {r['wall_clock_sec']:.1f}s |")
        lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PINN ablation sweeps.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--sweep", type=str, default="all", choices=["all", "optimizer", "loss_weights", "collocation"], help="Sweep selection")
    parser.add_argument("--fast", action="store_true", help="Fast smoke run with reduced iterations")
    parser.add_argument("--device", type=str, default=None, help="Compute device")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    device = args.device or cfg.get("training", {}).get("device", "cpu")
    figures_dir = cfg.get("paths", {}).get("figures_dir", "figures")
    results_dir = cfg.get("paths", {}).get("results_dir", "results")

    # Load ground truth benchmark
    benchmark_path = cfg.get("evaluation", {}).get("benchmark_path", "data/burgers_shock.mat")
    benchmark_url = cfg.get("evaluation", {}).get("benchmark_url", "https://github.com/maziarraissi/PINNs/raw/master/appendix/Data/burgers_shock.mat")
    data_file = ensure_benchmark_data(benchmark_path, benchmark_url)
    x_exact, t_exact, u_exact = load_benchmark(data_file)
    benchmark_data = (x_exact, t_exact, u_exact)

    opt_results = []
    weight_results = []
    colloc_results = []

    if args.sweep in ["all", "optimizer"]:
        opt_results = run_optimizer_sweep(cfg, benchmark_data, device=device, fast=args.fast)

    if args.sweep in ["all", "loss_weights"]:
        weight_results = run_loss_weight_sweep(cfg, benchmark_data, device=device, fast=args.fast)

    if args.sweep in ["all", "collocation"]:
        colloc_results = run_collocation_sweep(cfg, benchmark_data, device=device, fast=args.fast)

    # Save outputs
    json_path = os.path.join(results_dir, "ablations.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "optimizer_sweep": opt_results,
                "loss_weights_sweep": weight_results,
                "collocation_sweep": colloc_results,
            },
            f,
            indent=2,
        )
    print(f"\nAblations JSON saved to: {json_path}")

    plot_ablation_summaries(opt_results, weight_results, colloc_results, figures_dir)
    md_report = generate_markdown_report(opt_results, weight_results, colloc_results, os.path.join(results_dir, "ablation_tables.md"))
    print("\n" + md_report)


if __name__ == "__main__":
    main()
