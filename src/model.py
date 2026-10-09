"""BurgersPINN neural network architecture for 1D viscous Burgers' equation."""

from typing import Optional
import torch
import torch.nn as nn


class BurgersPINN(nn.Module):
    """Physics-Informed Neural Network (PINN) representing the scalar velocity u(x, t).

    Default architecture matches Raissi et al. (2019):
        - Input dimension: 2 (spatial coordinate x, temporal coordinate t)
        - 4 hidden layers with 50 units each
        - Tanh activation functions
        - Output dimension: 1 (predicted field u)
        - Total parameters: 7,851
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_dim: int = 50,
        hidden_layers: int = 4,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        act_fn = self._get_activation(activation)

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_fn())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    @staticmethod
    def _get_activation(act_name: str) -> type:
        act_lower = act_name.lower()
        if act_lower == "tanh":
            return nn.Tanh
        elif act_lower == "relu":
            return nn.ReLU
        elif act_lower == "gelu":
            return nn.GELU
        elif act_lower == "sin":
            class SineActivation(nn.Module):
                def forward(self, x):
                    return torch.sin(x)
            return SineActivation
        else:
            raise ValueError(f"Unsupported activation: {act_name}")

    def _init_weights(self) -> None:
        """Xavier normal initialization suitable for hyperbolic tangent PINNs."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass predicting u(x, t).

        Args:
            x: Tensor of spatial coordinates (N, 1), or concatenated (N, 2) when t is None.
            t: Tensor of temporal coordinates (N, 1), optional.

        Returns:
            Tensor of predicted solution u(x, t) of shape (N, 1).
        """
        if t is not None:
            inputs = torch.cat([x, t], dim=1)
        else:
            inputs = x
        return self.net(inputs)

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
