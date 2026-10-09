"""Burgers PINN Analysis package."""

from src.dataset import (
    BurgersDataset,
    create_burgers_dataset,
    create_eval_grid,
    sample_boundary,
    sample_collocation,
    sample_extrapolation,
    sample_initial,
)
from src.model import BurgersPINN
from src.physics import (
    LossBreakdown,
    compute_loss,
    compute_pde_loss,
    compute_pde_residual,
)
from src.trainer import PINNTrainer

__all__ = [
    "BurgersPINN",
    "BurgersDataset",
    "create_burgers_dataset",
    "create_eval_grid",
    "sample_collocation",
    "sample_boundary",
    "sample_initial",
    "sample_extrapolation",
    "compute_pde_residual",
    "compute_pde_loss",
    "compute_loss",
    "LossBreakdown",
    "PINNTrainer",
]
