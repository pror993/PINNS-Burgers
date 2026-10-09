"""Main CLI training script for Burgers PINN staged optimization."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from src.dataset import create_burgers_dataset
from src.model import BurgersPINN
from src.trainer import PINNTrainer


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_history(history: Dict[str, list], output_path: str) -> None:
    """Plot and save training convergence loss curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Stage 1: Adam
    if history.get("adam_epoch"):
        epochs = history["adam_epoch"]
        ax1.semilogy(epochs, history["adam_total_loss"], "b-", label="Total Loss", linewidth=1.5)
        ax1.semilogy(epochs, history["adam_loss_pde"], "g--", label="PDE Loss", linewidth=1.2)
        ax1.semilogy(epochs, history["adam_loss_ic"], "r:", label="IC Loss", linewidth=1.2)
        ax1.semilogy(epochs, history["adam_loss_bc"], "m-.", label="BC Loss", linewidth=1.2)
        ax1.set_title("Stage 1: Adam Convergence")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (log scale)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

    # Stage 2: L-BFGS
    if history.get("lbfgs_iter"):
        iters = history["lbfgs_iter"]
        ax2.semilogy(iters, history["lbfgs_total_loss"], "b-", label="Total Loss", linewidth=1.5)
        ax2.semilogy(iters, history["lbfgs_loss_pde"], "g--", label="PDE Loss", linewidth=1.2)
        ax2.semilogy(iters, history["lbfgs_loss_ic"], "r:", label="IC Loss", linewidth=1.2)
        ax2.semilogy(iters, history["lbfgs_loss_bc"], "m-.", label="BC Loss", linewidth=1.2)
        ax2.set_title("Stage 2: L-BFGS Refinement")
        ax2.set_xlabel("Evaluation Call")
        ax2.set_ylabel("Loss (log scale)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Training history curve saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Burgers PINN using staged optimization.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--adam-epochs", type=int, default=None, help="Override Adam epochs")
    parser.add_argument("--lbfgs-max-iter", type=int, default=None, help="Override L-BFGS max iterations")
    parser.add_argument("--adam-lr", type=float, default=None, help="Override Adam learning rate")
    parser.add_argument("--n-collocation", type=int, default=None, help="Override collocation point count")
    parser.add_argument("--device", type=str, default=None, help="Target device (cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--save-path", type=str, default=None, help="Override checkpoint destination path")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    # Resolve overrides
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    device = args.device if args.device is not None else cfg.get("training", {}).get("device", "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    adam_epochs = args.adam_epochs if args.adam_epochs is not None else cfg.get("training", {}).get("adam_epochs", 2000)
    adam_lr = args.adam_lr if args.adam_lr is not None else cfg.get("training", {}).get("adam_lr", 1e-3)
    adam_log_freq = cfg.get("training", {}).get("adam_log_freq", 200)
    lbfgs_max_iter = args.lbfgs_max_iter if args.lbfgs_max_iter is not None else cfg.get("training", {}).get("lbfgs_max_iter", 1000)
    lbfgs_tol_grad = cfg.get("training", {}).get("lbfgs_tolerance_grad", 1e-7)
    lbfgs_tol_change = cfg.get("training", {}).get("lbfgs_tolerance_change", 1e-9)
    lbfgs_hist_size = cfg.get("training", {}).get("lbfgs_history_size", 50)

    n_f = args.n_collocation if args.n_collocation is not None else cfg.get("collocation", {}).get("n_f", 2500)
    n_bc = cfg.get("collocation", {}).get("n_bc", 100)
    n_ic = cfg.get("collocation", {}).get("n_ic", 100)

    nu = cfg.get("physics", {}).get("nu", 0.01 / np.pi)
    x_bounds = (cfg.get("physics", {}).get("x_min", -1.0), cfg.get("physics", {}).get("x_max", 1.0))
    t_bounds = (cfg.get("physics", {}).get("t_min", 0.0), cfg.get("physics", {}).get("t_max", 1.0))

    w_ic = cfg.get("loss_weights", {}).get("w_ic", 50.0)
    w_bc = cfg.get("loss_weights", {}).get("w_bc", 10.0)
    w_pde = cfg.get("loss_weights", {}).get("w_pde", 1.0)

    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    checkpoint_file = cfg.get("paths", {}).get("checkpoint_file", "burgers_pinn.pt")
    save_path = args.save_path or os.path.join(checkpoint_dir, checkpoint_file)

    figures_dir = cfg.get("paths", {}).get("figures_dir", "figures")
    results_dir = cfg.get("paths", {}).get("results_dir", "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("==================================================")
    print("           Burgers PINN Training Script           ")
    print("==================================================")
    print(f"Device:               {device}")
    print(f"Seed:                 {seed}")
    print(f"Kinematic viscosity:  {nu:.6e}")
    print(f"Collocation points:   N_f={n_f}, N_bc={n_bc}, N_ic={n_ic}")
    print(f"Loss weights:         w_ic={w_ic}, w_bc={w_bc}, w_pde={w_pde}")
    print(f"Staged optimization:  Adam ({adam_epochs} epochs) -> L-BFGS ({lbfgs_max_iter} max iter)")
    print(f"Checkpoint target:    {save_path}")
    print("==================================================")

    set_seed(seed)

    # 1. Dataset Generation
    dataset = create_burgers_dataset(
        n_f=n_f,
        n_bc=n_bc,
        n_ic=n_ic,
        x_bounds=x_bounds,
        t_bounds=t_bounds,
        seed=seed,
    )

    # 2. Model Instantiation
    model_cfg = cfg.get("model", {})
    model = BurgersPINN(
        input_dim=model_cfg.get("input_dim", 2),
        output_dim=model_cfg.get("output_dim", 1),
        hidden_dim=model_cfg.get("hidden_dim", 50),
        hidden_layers=model_cfg.get("hidden_layers", 4),
        activation=model_cfg.get("activation", "tanh"),
    )
    print(f"Model parameters:     {model.count_parameters()}")

    # 3. Trainer Setup
    trainer = PINNTrainer(
        model=model,
        dataset=dataset,
        nu=nu,
        w_ic=w_ic,
        w_bc=w_bc,
        w_pde=w_pde,
        device=device,
    )

    # 4. Staged Training Execution
    start_total = time.time()
    train_results = trainer.train_staged(
        adam_epochs=adam_epochs,
        adam_lr=adam_lr,
        adam_log_freq=adam_log_freq,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_tolerance_grad=lbfgs_tol_grad,
        lbfgs_tolerance_change=lbfgs_tol_change,
        lbfgs_history_size=lbfgs_hist_size,
    )
    total_duration = time.time() - start_total

    # 5. Save Checkpoint
    trainer.save_checkpoint(
        save_path=save_path,
        extra_metadata={
            "config": cfg,
            "seed": seed,
            "training_results": train_results,
            "total_duration_sec": total_duration,
        },
    )

    # 6. Save Loss Curves
    loss_curve_path = os.path.join(figures_dir, "loss_curve.png")
    plot_training_history(trainer.history, loss_curve_path)

    # 7. Write Summary JSON
    summary_path = os.path.join(results_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "duration_sec": total_duration,
                "model_parameters": model.count_parameters(),
                "final_loss": train_results["final_loss"],
                "checkpoint": save_path,
            },
            f,
            indent=2,
        )
    print(f"Training summary saved to {summary_path}")
    print("Training finished successfully.")


if __name__ == "__main__":
    main()
