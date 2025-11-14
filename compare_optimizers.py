"""
Optimizer Comparison Script

Compares Deep Momentum GD with standard optimizers (SGD+Momentum, Adam)
on both 2D optimization and neural network training tasks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import nested learning optimizer
from nested_learning import DeepMomentumGD

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

print("=" * 70)
print("OPTIMIZER COMPARISON EXPERIMENTS")
print("=" * 70)


def rosenbrock(x, y, a=1, b=100):
    """Rosenbrock function - classic optimization test function"""
    return (a - x)**2 + b * (y - x**2)**2


def rosenbrock_gradient(x, y, a=1, b=100):
    """Gradient of Rosenbrock function"""
    dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dy = 2 * b * (y - x**2)
    return torch.tensor([dx, dy], dtype=torch.float32)


def optimize_2d_function(optimizer_name, optimizer, initial_point, num_steps=200):
    """
    Optimize a 2D function and track the trajectory

    Args:
        optimizer_name: Name of the optimizer
        optimizer: PyTorch optimizer
        initial_point: Starting point [x, y]
        num_steps: Number of optimization steps

    Returns:
        trajectory: List of (x, y) points
        losses: List of function values
    """
    # Create parameter
    param = torch.tensor(initial_point, dtype=torch.float32, requires_grad=True)

    # Setup optimizer
    if optimizer_name == "DMGD":
        opt = optimizer([param])
    else:
        opt = optimizer([param])

    trajectory = [param.detach().cpu().numpy().copy()]
    losses = []

    for step in range(num_steps):
        opt.zero_grad()

        # Compute loss
        x, y = param[0], param[1]
        loss = rosenbrock(x, y)
        losses.append(loss.item())

        # Backward pass
        loss.backward()

        # Optimization step
        opt.step()

        # Record trajectory
        trajectory.append(param.detach().cpu().numpy().copy())

    return np.array(trajectory), np.array(losses)


def plot_2d_optimization():
    """Compare optimizers on 2D Rosenbrock function"""
    print("\n" + "=" * 70)
    print("Experiment 1: 2D Optimization (Rosenbrock Function)")
    print("=" * 70)

    # Setup
    initial_point = [-1.0, 1.0]
    num_steps = 200

    # Define optimizers
    optimizers = {
        "DMGD": lambda params: DeepMomentumGD(
            params, lr=0.01, momentum=0.9,
            memory_depth=2, memory_hidden=32
        ),
        "SGD+Momentum": lambda params: torch.optim.SGD(
            params, lr=0.01, momentum=0.9
        ),
        "Adam": lambda params: torch.optim.Adam(
            params, lr=0.01
        ),
    }

    # Run optimization for each optimizer
    results = {}
    for name, opt_fn in optimizers.items():
        print(f"\nRunning {name}...")
        trajectory, losses = optimize_2d_function(name, opt_fn, initial_point, num_steps)
        results[name] = {"trajectory": trajectory, "losses": losses}
        final_loss = losses[-1]
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Reduction: {losses[0] - final_loss:.4f}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Trajectories on contour plot
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    # Contour plot
    levels = np.logspace(-1, 3.5, 20)
    ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.1)

    # Plot trajectories
    colors = {'DMGD': 'red', 'SGD+Momentum': 'blue', 'Adam': 'green'}
    for name, data in results.items():
        traj = data['trajectory']
        ax1.plot(traj[:, 0], traj[:, 1], '-o',
                label=name, color=colors[name],
                markersize=3, linewidth=2, alpha=0.7)
        # Mark start and end
        ax1.plot(traj[0, 0], traj[0, 1], 'ko', markersize=10, label='Start' if name == 'DMGD' else '')
        ax1.plot(traj[-1, 0], traj[-1, 1], 'k*', markersize=15, label='End' if name == 'DMGD' else '')

    # Mark optimum
    ax1.plot(1, 1, 'r*', markersize=20, label='Optimum (1,1)', zorder=10)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Optimizer Trajectories on Rosenbrock Function', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss curves
    for name, data in results.items():
        ax2.semilogy(data['losses'], label=name,
                    color=colors[name], linewidth=2)

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = results_dir / "optimizer_comparison_2d.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved 2D comparison plot to: {save_path}")

    plt.close()

    return results


def create_regression_dataset(n_samples=1000, n_features=20, noise=0.1):
    """Create a synthetic regression dataset"""
    X = torch.randn(n_samples, n_features)
    true_weights = torch.randn(n_features, 1)
    y = X @ true_weights + noise * torch.randn(n_samples, 1)
    return X, y, true_weights


def train_neural_network(optimizer_fn, X_train, y_train, X_test, y_test,
                         num_epochs=50, hidden_dim=64):
    """
    Train a neural network with given optimizer

    Returns:
        train_losses: Training loss history
        test_losses: Test loss history
    """
    # Create model
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )

    # Create optimizer
    optimizer = optimizer_fn(model.parameters())

    # Training loop
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        train_pred = model(X_train)
        train_loss = nn.functional.mse_loss(train_pred, y_train)

        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = nn.functional.mse_loss(test_pred, y_test)
            test_losses.append(test_loss.item())

    return train_losses, test_losses


def plot_neural_network_training():
    """Compare optimizers on neural network training"""
    print("\n" + "=" * 70)
    print("Experiment 2: Neural Network Training (Regression Task)")
    print("=" * 70)

    # Create dataset
    print("\nGenerating synthetic regression dataset...")
    X, y, true_weights = create_regression_dataset(n_samples=1000, n_features=20)

    # Split train/test
    split = 800
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")

    # Define optimizers
    optimizers = {
        "DMGD": lambda params: DeepMomentumGD(
            params, lr=0.01, momentum=0.9,
            memory_depth=2, memory_hidden=32
        ),
        "SGD+Momentum": lambda params: torch.optim.SGD(
            params, lr=0.01, momentum=0.9
        ),
        "Adam": lambda params: torch.optim.Adam(
            params, lr=0.01
        ),
    }

    # Train with each optimizer
    results = {}
    for name, opt_fn in optimizers.items():
        print(f"\nTraining with {name}...")
        train_losses, test_losses = train_neural_network(
            opt_fn, X_train, y_train, X_test, y_test, num_epochs=50
        )
        results[name] = {
            "train_losses": train_losses,
            "test_losses": test_losses
        }
        print(f"  Final train loss: {train_losses[-1]:.6f}")
        print(f"  Final test loss: {test_losses[-1]:.6f}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {'DMGD': 'red', 'SGD+Momentum': 'blue', 'Adam': 'green'}

    # Plot training losses
    for name, data in results.items():
        ax1.semilogy(data['train_losses'], label=name,
                    color=colors[name], linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss (log scale)', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot test losses
    for name, data in results.items():
        ax2.semilogy(data['test_losses'], label=name,
                    color=colors[name], linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Loss (log scale)', fontsize=12)
    ax2.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = results_dir / "optimizer_comparison_nn.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved neural network comparison plot to: {save_path}")

    plt.close()

    return results


def main():
    """Run all comparison experiments"""

    # Run experiments
    print("\nStarting comparison experiments...")

    # Experiment 1: 2D Optimization
    results_2d = plot_2d_optimization()

    # Experiment 2: Neural Network Training
    results_nn = plot_neural_network_training()

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print("\n2D Optimization (Rosenbrock):")
    for name, data in results_2d.items():
        final_loss = data['losses'][-1]
        print(f"  {name:15s} - Final loss: {final_loss:.6f}")

    print("\nNeural Network Training:")
    for name, data in results_nn.items():
        final_test_loss = data['test_losses'][-1]
        print(f"  {name:15s} - Final test loss: {final_test_loss:.6f}")

    print("\n" + "=" * 70)
    print("✓ All experiments completed successfully!")
    print(f"✓ Results saved to: {results_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
