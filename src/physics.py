"""Autograd PDE residual engine and loss functions for Burgers' equation."""

from dataclasses import dataclass
from typing import Tuple, Union
import torch
import torch.nn as nn

from src.dataset import BurgersDataset


@dataclass
class LossBreakdown:
    """Detailed components of the PINN composite loss."""

    total: torch.Tensor
    ic: torch.Tensor
    bc: torch.Tensor
    pde: torch.Tensor

    def to_dict(self) -> dict:
        return {
            "total_loss": self.total.item(),
            "loss_ic": self.ic.item(),
            "loss_bc": self.bc.item(),
            "loss_pde": self.pde.item(),
        }


def compute_pde_residual(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Compute the governing Burgers PDE residual: R = u_t + u*u_x - nu*u_xx.

    Args:
        model: PINN neural network model mapping (x, t) -> u.
        x: Spatial collocation tensor with requires_grad=True, shape (N, 1).
        t: Temporal collocation tensor with requires_grad=True, shape (N, 1).
        nu: Kinematic viscosity coefficient (default: 0.01 / pi).

    Returns:
        Tensor of PDE residuals R of shape (N, 1).
    """
    u = model(x, t)

    # First-order temporal derivative u_t
    u_t = torch.autograd.grad(
        outputs=u,
        inputs=t,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]

    # First-order spatial derivative u_x
    u_x = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Second-order spatial derivative u_xx
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Burgers' residual R = u_t + u * u_x - nu * u_xx
    residual = u_t + u * u_x - nu * u_xx
    return residual


def compute_pde_loss(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Compute mean squared PDE residual over collocation points."""
    residual = compute_pde_residual(model, x, t, nu)
    return torch.mean(residual ** 2)


def compute_loss(
    model: nn.Module,
    dataset: BurgersDataset,
    nu: float,
    w_ic: float = 50.0,
    w_bc: float = 10.0,
    w_pde: float = 1.0,
) -> LossBreakdown:
    """Compute weighted composite loss: L = w_ic*L_ic + w_bc*L_bc + w_pde*L_pde.

    Args:
        model: Neural network model.
        dataset: BurgersDataset holding collocation, IC, and BC tensors.
        nu: Viscosity parameter.
        w_ic: Weight for initial condition loss (default 50.0).
        w_bc: Weight for boundary condition loss (default 10.0).
        w_pde: Weight for PDE residual loss (default 1.0).

    Returns:
        LossBreakdown dataclass with total, ic, bc, and pde loss tensors.
    """
    # Initial Condition Loss (t = 0)
    u_in_pred = model(dataset.x_in, dataset.t_in)
    loss_ic = torch.mean((u_in_pred - dataset.u_in) ** 2)

    # Boundary Condition Loss (x = -1, x = 1)
    u_b_pred = model(dataset.x_b, dataset.t_b)
    loss_bc = torch.mean((u_b_pred - dataset.u_b) ** 2)

    # Collocation PDE Residual Loss
    loss_pde = compute_pde_loss(model, dataset.x_f, dataset.t_f, nu)

    total_loss = w_ic * loss_ic + w_bc * loss_bc + w_pde * loss_pde

    return LossBreakdown(
        total=total_loss,
        ic=loss_ic,
        bc=loss_bc,
        pde=loss_pde,
    )
