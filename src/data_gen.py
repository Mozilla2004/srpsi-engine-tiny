"""
Data Generation for 1D Field Evolution Tasks
==================================================

Generates synthetic datasets for:
- 1D Burgers equation: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
- Nonlinear wave equation (future)

Author: SRΨ-Engine Tiny Experiment
"""

import numpy as np
from typing import Tuple
import argparse


def random_initial_condition(nx: int, seed=None) -> np.ndarray:
    """
    Generate random initial condition for 1D field.

    Combines:
    - Multi-frequency sinusoidal components
    - Optional Gaussian bump

    Args:
        nx: Spatial resolution
        seed: Random seed for reproducibility

    Returns:
        u0: Initial field [nx]
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)

    # Random sinusoidal component
    amp = np.random.uniform(0.5, 1.5)
    freq = np.random.randint(1, 4)
    phase = np.random.uniform(0, 2 * np.pi)

    signal = amp * np.sin(freq * x + phase)

    # Optional Gaussian bump
    if np.random.rand() > 0.5:
        center = np.random.uniform(0, 2 * np.pi)
        width = np.random.uniform(0.2, 0.8)
        bump = 0.5 * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        u0 = signal + bump
    else:
        u0 = signal

    return u0.astype(np.float32)


def burgers_step(u: np.ndarray, dt: float, dx: float, nu: float = 0.01) -> np.ndarray:
    """
    Single time step of 1D Burgers equation using finite differences.
    Uses sub-stepping for numerical stability.

    Scheme: Forward Euler in time, central difference in space
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    """
    # Use sub-steps for stability
    n_substeps = 10
    dt_sub = dt / n_substeps

    for _ in range(n_substeps):
        # Periodic boundary conditions using np.roll
        u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)      # First derivative (central)
        u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx ** 2)  # Second derivative

        # Burgers equation: du/dt = -u·du/dx + ν·d²u/dx²
        du_dt = -u * u_x + nu * u_xx
        u = u + dt_sub * du_dt

        # Safety: clip values to prevent explosion
        u = np.clip(u, -10.0, 10.0)

    return u.astype(np.float32)


def generate_burgers_1d(
    num_samples: int,
    total_steps: int,
    nx: int,
    dt: float,
    dx: float,
    nu: float = 0.01,
    seed: int = 42
) -> np.ndarray:
    """
    Generate dataset of 1D Burgers equation trajectories.

    Output shape: [num_samples, total_steps, nx]

    Args:
        num_samples: Number of trajectories to generate
        total_steps: Total number of time steps per trajectory
        nx: Spatial resolution
        dt: Time step size
        dx: Spatial step size
        nu: Viscosity coefficient
        seed: Random seed

    Returns:
        data: Generated trajectories [num_samples, total_steps, nx]
    """
    data = np.zeros((num_samples, total_steps, nx), dtype=np.float32)

    for i in range(num_samples):
        # Random initial condition
        u = random_initial_condition(nx, seed=seed + i)

        # Time integration
        for t in range(total_steps):
            data[i, t] = u
            u = burgers_step(u, dt, dx, nu)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")

    return data


def generate_nonlinear_wave_1d(
    num_samples: int,
    total_steps: int,
    nx: int,
    dt: float,
    dx: float,
    seed: int = 42
) -> np.ndarray:
    """
    Generate dataset of 1D nonlinear wave equation (placeholder).

    TODO: Implement nonlinear wave: ∂²u/∂t² = c²·∂²u/∂x² + nonlinear_term

    Args:
        num_samples: Number of trajectories
        total_steps: Total time steps
        nx: Spatial resolution
        dt: Time step
        dx: Spatial step
        seed: Random seed

    Returns:
        data: Generated trajectories [num_samples, total_steps, nx]
    """
    raise NotImplementedError("Nonlinear wave generator not implemented yet")


def main():
    """CLI for data generation"""
    parser = argparse.ArgumentParser(description="Generate synthetic field evolution data")
    parser.add_argument("--task", type=str, default="burgers_1d", choices=["burgers_1d", "wave_1d"])
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--total_steps", type=int, default=48)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="data/burgers_1d.npy")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(f"Generating {args.task} dataset...")
    print(f"  Samples: {args.num_samples}")
    print(f"  Steps: {args.total_steps}")
    print(f"  Grid: {args.nx} x {args.total_steps}")

    if args.task == "burgers_1d":
        # Calculate dx if not explicitly set to match domain [0, 2π]
        domain_size = 2 * np.pi
        dx = domain_size / args.nx
        print(f"  Calculated dx: {dx:.6f} (matching 2π domain)")

        data = generate_burgers_1d(
            num_samples=args.num_samples,
            total_steps=args.total_steps,
            nx=args.nx,
            dt=args.dt,
            dx=dx,
            nu=args.nu,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Save dataset
    np.save(args.output, data)
    print(f"\nDataset saved to: {args.output}")
    print(f"Shape: {data.shape}")
    print(f"Size: {data.nbytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
