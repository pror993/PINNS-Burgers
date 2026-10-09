"""Staged optimization engine (Adam + L-BFGS) for BurgersPINN training."""

import os
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from src.dataset import BurgersDataset
from src.physics import LossBreakdown, compute_loss


class PINNTrainer:
    """Trainer coordinating staged Adam + L-BFGS optimization for PINNs."""

    def __init__(
        self,
        model: nn.Module,
        dataset: BurgersDataset,
        nu: float,
        w_ic: float = 50.0,
        w_bc: float = 10.0,
        w_pde: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.dataset = dataset.to(self.device)
        self.nu = nu
        self.w_ic = w_ic
        self.w_bc = w_bc
        self.w_pde = w_pde

        self.history: Dict[str, List[float]] = {
            "adam_epoch": [],
            "adam_total_loss": [],
            "adam_loss_ic": [],
            "adam_loss_bc": [],
            "adam_loss_pde": [],
            "lbfgs_iter": [],
            "lbfgs_total_loss": [],
            "lbfgs_loss_ic": [],
            "lbfgs_loss_bc": [],
            "lbfgs_loss_pde": [],
        }

    def compute_current_loss(self) -> LossBreakdown:
        """Compute the current loss on the training dataset."""
        return compute_loss(
            model=self.model,
            dataset=self.dataset,
            nu=self.nu,
            w_ic=self.w_ic,
            w_bc=self.w_bc,
            w_pde=self.w_pde,
        )

    def train_adam(
        self,
        epochs: int = 2000,
        lr: float = 1e-3,
        log_freq: int = 200,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Stage 1: First-order Adam optimization to locate the basin of attraction."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        start_time = time.time()

        if verbose:
            print(f"--- Starting Adam Optimization ({epochs} epochs, lr={lr}) ---")

        for epoch in range(epochs):
            optimizer.zero_grad()
            breakdown = self.compute_current_loss()
            breakdown.total.backward()
            optimizer.step()

            # Record history
            if (epoch % log_freq == 0) or (epoch == epochs - 1):
                self.history["adam_epoch"].append(epoch)
                self.history["adam_total_loss"].append(breakdown.total.item())
                self.history["adam_loss_ic"].append(breakdown.ic.item())
                self.history["adam_loss_bc"].append(breakdown.bc.item())
                self.history["adam_loss_pde"].append(breakdown.pde.item())

                if verbose and (epoch % log_freq == 0 or epoch == epochs - 1):
                    print(
                        f"Adam Epoch {epoch:5d} | "
                        f"Total: {breakdown.total.item():.4e} | "
                        f"IC: {breakdown.ic.item():.4e} | "
                        f"BC: {breakdown.bc.item():.4e} | "
                        f"PDE: {breakdown.pde.item():.4e}"
                    )

        elapsed = time.time() - start_time
        final_loss = self.compute_current_loss()
        if verbose:
            print(f"Adam completed in {elapsed:.2f}s | Final Total Loss: {final_loss.total.item():.4e}")

        return {
            "elapsed_sec": elapsed,
            "final_total_loss": final_loss.total.item(),
            "final_loss_pde": final_loss.pde.item(),
        }

    def train_lbfgs(
        self,
        max_iter: int = 1000,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 50,
        log_freq: int = 100,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Stage 2: Second-order quasi-Newton L-BFGS optimization for sharp gradient convergence."""
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
        )

        iter_counter = {"count": 0}
        start_time = time.time()

        if verbose:
            print(f"--- Starting L-BFGS Optimization (max_iter={max_iter}) ---")

        def closure():
            optimizer.zero_grad()
            breakdown = self.compute_current_loss()
            breakdown.total.backward()

            count = iter_counter["count"]
            if (count % log_freq == 0) or (count == max_iter - 1):
                self.history["lbfgs_iter"].append(count)
                self.history["lbfgs_total_loss"].append(breakdown.total.item())
                self.history["lbfgs_loss_ic"].append(breakdown.ic.item())
                self.history["lbfgs_loss_bc"].append(breakdown.bc.item())
                self.history["lbfgs_loss_pde"].append(breakdown.pde.item())

                if verbose:
                    print(
                        f"L-BFGS Iter {count:4d} | "
                        f"Total: {breakdown.total.item():.4e} | "
                        f"IC: {breakdown.ic.item():.4e} | "
                        f"BC: {breakdown.bc.item():.4e} | "
                        f"PDE: {breakdown.pde.item():.4e}"
                    )

            iter_counter["count"] += 1
            return breakdown.total

        optimizer.step(closure)
        elapsed = time.time() - start_time

        post_loss = self.compute_current_loss()
        if verbose:
            print(
                f"L-BFGS completed in {elapsed:.2f}s ({iter_counter['count']} calls) | "
                f"Post L-BFGS | Total: {post_loss.total.item():.4e} | "
                f"IC: {post_loss.ic.item():.4e} | "
                f"BC: {post_loss.bc.item():.4e} | "
                f"PDE: {post_loss.pde.item():.4e}"
            )

        return {
            "elapsed_sec": elapsed,
            "iterations": iter_counter["count"],
            "post_total_loss": post_loss.total.item(),
            "post_loss_ic": post_loss.ic.item(),
            "post_loss_bc": post_loss.bc.item(),
            "post_loss_pde": post_loss.pde.item(),
        }

    def train_staged(
        self,
        adam_epochs: int = 2000,
        adam_lr: float = 1e-3,
        adam_log_freq: int = 200,
        lbfgs_max_iter: int = 1000,
        lbfgs_tolerance_grad: float = 1e-7,
        lbfgs_tolerance_change: float = 1e-9,
        lbfgs_history_size: int = 50,
        lbfgs_log_freq: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute the complete two-stage optimization strategy."""
        start_time = time.time()

        adam_res = self.train_adam(
            epochs=adam_epochs,
            lr=adam_lr,
            log_freq=adam_log_freq,
            verbose=verbose,
        )

        lbfgs_res = self.train_lbfgs(
            max_iter=lbfgs_max_iter,
            tolerance_grad=lbfgs_tolerance_grad,
            tolerance_change=lbfgs_tolerance_change,
            history_size=lbfgs_history_size,
            log_freq=lbfgs_log_freq,
            verbose=verbose,
        )

        total_elapsed = time.time() - start_time
        final_loss = self.compute_current_loss()

        return {
            "total_elapsed_sec": total_elapsed,
            "adam": adam_res,
            "lbfgs": lbfgs_res,
            "final_loss": final_loss.to_dict(),
        }

    def save_checkpoint(
        self,
        save_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model weights, training history, and metadata to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "nu": self.nu,
            "w_ic": self.w_ic,
            "w_bc": self.w_bc,
            "w_pde": self.w_pde,
            "history": self.history,
            "extra_metadata": extra_metadata or {},
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model state and metadata from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.nu = checkpoint.get("nu", self.nu)
        self.w_ic = checkpoint.get("w_ic", self.w_ic)
        self.w_bc = checkpoint.get("w_bc", self.w_bc)
        self.w_pde = checkpoint.get("w_pde", self.w_pde)
        self.history = checkpoint.get("history", self.history)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
