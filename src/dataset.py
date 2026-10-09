"""Latin Hypercube Sampling routines for Burgers PINN training and evaluation."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.stats import qmc
import torch


@dataclass
class BurgersDataset:
    """Container holding collocation, initial, and boundary condition tensors."""

    x_f: torch.Tensor   # Collocation spatial coordinates (N_f, 1) [requires_grad=True]
    t_f: torch.Tensor   # Collocation temporal coordinates (N_f, 1) [requires_grad=True]
    x_b: torch.Tensor   # Boundary spatial coordinates (N_bc, 1)
    t_b: torch.Tensor   # Boundary temporal coordinates (N_bc, 1)
    u_b: torch.Tensor   # Boundary target values (N_bc, 1)
    x_in: torch.Tensor  # Initial spatial coordinates (N_ic, 1)
    t_in: torch.Tensor  # Initial temporal coordinates (N_ic, 1)
    u_in: torch.Tensor  # Initial target values (N_ic, 1)

    def to(self, device: torch.device | str) -> "BurgersDataset":
        """Move all tensors to target device and ensure collocation gradient tracking."""
        x_f = self.x_f.to(device).detach().requires_grad_(True)
        t_f = self.t_f.to(device).detach().requires_grad_(True)
        return BurgersDataset(
            x_f=x_f,
            t_f=t_f,
            x_b=self.x_b.to(device),
            t_b=self.t_b.to(device),
            u_b=self.u_b.to(device),
            x_in=self.x_in.to(device),
            t_in=self.t_in.to(device),
            u_in=self.u_in.to(device),
        )


def sample_collocation(
    n_f: int = 2500,
    x_bounds: Tuple[float, float] = (-1.0, 1.0),
    t_bounds: Tuple[float, float] = (0.0, 1.0),
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample 2D Latin Hypercube collocation points inside the spatio-temporal domain.

    Returns:
        x_f, t_f: Tensors of shape (n_f, 1) with requires_grad=True.
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    raw_samples = sampler.random(n=n_f)
    scaled = qmc.scale(
        raw_samples,
        l_bounds=[x_bounds[0], t_bounds[0]],
        u_bounds=[x_bounds[1], t_bounds[1]],
    )
    x_f = torch.tensor(scaled[:, 0:1], dtype=torch.float32, requires_grad=True)
    t_f = torch.tensor(scaled[:, 1:2], dtype=torch.float32, requires_grad=True)
    return x_f, t_f


def sample_boundary(
    n_bc: int = 100,
    t_bounds: Tuple[float, float] = (0.0, 1.0),
    x_left_val: float = -1.0,
    x_right_val: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample 1D boundary points along x = -1 and x = +1 across t in [0, 1].

    Returns:
        x_b: Spatial coordinates of shape (n_bc, 1).
        t_b: Temporal coordinates of shape (n_bc, 1).
        u_b: Target boundary velocities of shape (n_bc, 1) [all zeros].
    """
    sampler_1d = qmc.LatinHypercube(d=1, seed=seed)
    t_samples = qmc.scale(sampler_1d.random(n=n_bc), t_bounds[0], t_bounds[1])
    t_b = torch.tensor(t_samples, dtype=torch.float32)

    n_left = n_bc // 2
    n_right = n_bc - n_left
    x_left = torch.full((n_left, 1), x_left_val, dtype=torch.float32)
    x_right = torch.full((n_right, 1), x_right_val, dtype=torch.float32)
    x_b = torch.vstack([x_left, x_right])
    u_b = torch.zeros_like(x_b)

    return x_b, t_b, u_b


def sample_initial(
    n_ic: int = 100,
    x_bounds: Tuple[float, float] = (-1.0, 1.0),
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample 1D initial condition points along t = 0 for x in [-1, 1].

    Target velocity profile is u(x, 0) = -sin(pi * x).

    Returns:
        x_in: Spatial coordinates of shape (n_ic, 1).
        t_in: Temporal coordinates (all zeros) of shape (n_ic, 1).
        u_in: Target initial velocities of shape (n_ic, 1).
    """
    sampler_1d = qmc.LatinHypercube(d=1, seed=seed)
    x_samples = qmc.scale(sampler_1d.random(n=n_ic), x_bounds[0], x_bounds[1])
    t_in = torch.zeros(n_ic, 1, dtype=torch.float32)
    x_in = torch.tensor(x_samples, dtype=torch.float32)
    u_in = -torch.sin(torch.pi * x_in)

    return x_in, t_in, u_in


def sample_extrapolation(
    n_ext: int = 1000,
    x_bounds: Tuple[float, float] = (-1.0, 1.0),
    t_ext_bounds: Tuple[float, float] = (1.0, 1.3),
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample unseen future points outside the training domain (t > 1.0).

    Returns:
        x_ext, t_ext: Tensors of shape (n_ext, 1) with requires_grad=True.
    """
    sampler_ext = qmc.LatinHypercube(d=2, seed=seed)
    scaled_ext = qmc.scale(
        sampler_ext.random(n=n_ext),
        l_bounds=[x_bounds[0], t_ext_bounds[0]],
        u_bounds=[x_bounds[1], t_ext_bounds[1]],
    )
    x_ext = torch.tensor(scaled_ext[:, 0:1], dtype=torch.float32, requires_grad=True)
    t_ext = torch.tensor(scaled_ext[:, 1:2], dtype=torch.float32, requires_grad=True)
    return x_ext, t_ext


def create_burgers_dataset(
    n_f: int = 2500,
    n_bc: int = 100,
    n_ic: int = 100,
    x_bounds: Tuple[float, float] = (-1.0, 1.0),
    t_bounds: Tuple[float, float] = (0.0, 1.0),
    seed: Optional[int] = None,
) -> BurgersDataset:
    """Build the complete Burgers PINN dataset according to notebook specification."""
    seed_f = seed if seed is None else seed
    seed_bc = seed if seed is None else seed + 1
    seed_ic = seed if seed is None else seed + 2

    x_f, t_f = sample_collocation(n_f=n_f, x_bounds=x_bounds, t_bounds=t_bounds, seed=seed_f)
    x_b, t_b, u_b = sample_boundary(n_bc=n_bc, t_bounds=t_bounds, seed=seed_bc)
    x_in, t_in, u_in = sample_initial(n_ic=n_ic, x_bounds=x_bounds, seed=seed_ic)

    return BurgersDataset(
        x_f=x_f,
        t_f=t_f,
        x_b=x_b,
        t_b=t_b,
        u_b=u_b,
        x_in=x_in,
        t_in=t_in,
        u_in=u_in,
    )


def create_eval_grid(
    n_x: int = 256,
    n_t: int = 100,
    x_bounds: Tuple[float, float] = (-1.0, 1.0),
    t_bounds: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Generate regular rectangular grid for spatial-temporal evaluation.

    Returns:
        X, T: 2D numpy arrays of shape (n_t, n_x).
        x_eval, t_eval: Flattened torch tensors of shape (n_t * n_x, 1).
    """
    x_grid = np.linspace(x_bounds[0], x_bounds[1], n_x)
    t_grid = np.linspace(t_bounds[0], t_bounds[1], n_t)
    X, T = np.meshgrid(x_grid, t_grid)

    x_eval = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    t_eval = torch.tensor(T.flatten()[:, None], dtype=torch.float32)

    return X, T, x_eval, t_eval
