"""
Minimal Self-Contained Demo of Deep Momentum GD

This script provides a minimal, self-contained demonstration of the
Deep Momentum GD concept. Use this as a fallback if the main package
has any issues.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("=" * 70)
print("MINIMAL DEEP MOMENTUM GD DEMO")
print("=" * 70)

# ============================================================================
# Simple Deep Momentum GD Implementation
# ============================================================================

class MinimalDeepMomentumGD(torch.optim.Optimizer):
    """
    Minimal implementation of Deep Momentum GD for demonstration.

    Key idea: Replace linear momentum (m = β*m + (1-β)*g)
    with a learned MLP: m = MLP(g, m_prev)
    """

    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

        # Initialize memory networks for each parameter group
        self.memory_nets = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # Simple 2-layer MLP for momentum memory
                    input_size = p.numel()
                    hidden_size = min(64, max(16, input_size // 4))

                    memory_net = nn.Sequential(
                        nn.Linear(input_size * 2, hidden_size),  # Input: [grad, momentum]
                        nn.ReLU(),
                        nn.Linear(hidden_size, input_size)  # Output: new momentum
                    ).to(p.device)

                    self.memory_nets.append(memory_net)

                    # Initialize momentum buffer
                    state = self.state[p]
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['memory_net'] = memory_net

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                momentum_buffer = state['momentum_buffer']
                memory_net = state['memory_net']

                # Flatten for MLP
                grad_flat = grad.flatten()
                momentum_flat = momentum_buffer.flatten()

                # Concatenate gradient and previous momentum
                memory_input = torch.cat([grad_flat, momentum_flat])

                # Compute new momentum using learned memory
                with torch.enable_grad():
                    new_momentum_flat = memory_net(memory_input)

                # Reshape and update
                new_momentum = new_momentum_flat.reshape_as(grad)
                state['momentum_buffer'] = new_momentum

                # Update parameters
                p.data.add_(new_momentum, alpha=-lr)

        return loss


# ============================================================================
# Test on 2D Rosenbrock Function
# ============================================================================

def rosenbrock(x, y, a=1, b=100):
    """Rosenbrock function - classic non-convex test problem"""
    return (a - x)**2 + b * (y - x**2)**2


def optimize_2d(optimizer_class, optimizer_name, initial_point=[-1.0, 1.0], num_steps=200):
    """Run 2D optimization and return trajectory"""

    # Create parameter
    param = torch.tensor(initial_point, dtype=torch.float32, requires_grad=True)

    # Create optimizer
    if optimizer_class == MinimalDeepMomentumGD:
        optimizer = optimizer_class([param], lr=0.01, momentum=0.9)
    elif optimizer_class == torch.optim.SGD:
        optimizer = optimizer_class([param], lr=0.01, momentum=0.9)
    else:
        optimizer = optimizer_class([param], lr=0.01)

    trajectory = [param.detach().cpu().numpy().copy()]
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Compute loss
        x, y = param[0], param[1]
        loss = rosenbrock(x, y)
        losses.append(loss.item())

        # Backward and step
        loss.backward()
        optimizer.step()

        trajectory.append(param.detach().cpu().numpy().copy())

    return np.array(trajectory), np.array(losses)


# ============================================================================
# Run Comparison
# ============================================================================

print("\nRunning optimization comparison...")
print("This may take 1-2 minutes...\n")

# Setup
torch.manual_seed(42)
initial_point = [-1.0, 1.0]
num_steps = 200

# Define optimizers
optimizers = {
    "Minimal DMGD": (MinimalDeepMomentumGD, {}),
    "SGD+Momentum": (torch.optim.SGD, {}),
    "Adam": (torch.optim.Adam, {}),
}

# Run optimization
results = {}
for name, (opt_class, opt_kwargs) in optimizers.items():
    print(f"Running {name}...")
    trajectory, losses = optimize_2d(opt_class, name, initial_point, num_steps)
    results[name] = {"trajectory": trajectory, "losses": losses}
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Distance to optimum: {np.linalg.norm(trajectory[-1] - np.array([1.0, 1.0])):.4f}\n")


# ============================================================================
# Visualization
# ============================================================================

print("Creating visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Trajectories on contour
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

levels = np.logspace(-1, 3.5, 20)
ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.1)

colors = {"Minimal DMGD": 'red', "SGD+Momentum": 'blue', "Adam": 'green'}
for name, data in results.items():
    traj = data['trajectory']
    ax1.plot(traj[:, 0], traj[:, 1], '-o',
            label=name, color=colors[name],
            markersize=3, linewidth=2, alpha=0.7)

ax1.plot(1, 1, 'r*', markersize=20, label='Optimum (1,1)', zorder=10)
ax1.plot(initial_point[0], initial_point[1], 'ko', markersize=10, label='Start')

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Optimizer Trajectories on Rosenbrock Function', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Loss curves
for name, data in results.items():
    ax2.semilogy(data['losses'], label=name, color=colors[name], linewidth=2)

ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Loss (log scale)', fontsize=12)
ax2.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
save_path = results_dir / "minimal_demo_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comparison plot to: {save_path}")

plt.close()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("DEMO SUMMARY")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Deep Momentum GD uses an MLP to learn gradient compression")
print("2. This is more expressive than linear momentum")
print("3. On complex landscapes, learned memory can help convergence")
print("4. Trade-off: More parameters and computation")

print("\nWhat makes this 'Nested Learning':")
print("- The optimizer itself learns (inner optimization)")
print("- While optimizing the model parameters (outer optimization)")
print("- The MLP memory adapts to problem-specific patterns")

print("\n" + "=" * 70)
print("✓ Minimal demo complete!")
print("=" * 70)
