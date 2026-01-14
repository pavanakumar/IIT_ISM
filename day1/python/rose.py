#!/usr/bin/env python3
"""
=============================================================================
Exercise 1: Rosenbrock Function Optimization
=============================================================================

This hands-on exercise demonstrates:
  1. 3D visualization of the Rosenbrock function
  2. Gradient-free optimization using CMA-ES
  3. Gradient-based optimization using L-BFGS with PyTorch autograd
  4. Finite difference gradient estimation and error analysis

The Rosenbrock function is defined as:
    f(x, y) = (a - x)^2 + b * (y - x^2)^2

With parameters a=1, b=100, the global minimum is at (1, 1) with f(1,1) = 0.

Learning objectives:
  - Understand the difference between gradient-free and gradient-based methods
  - See how PyTorch's autograd computes exact gradients automatically
  - Analyze the accuracy of finite difference approximations

=============================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.optimize import minimize
from cma import CMAEvolutionStrategy
import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available (optional, CPU works fine for this exercise)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# SECTION 1: Define the Rosenbrock Function
# =============================================================================

def rosenbrock_numpy(x, params):
    """
    Rosenbrock function for NumPy arrays.

    Args:
        x: Array of shape (2,) containing [x, y] coordinates
        params: List [a, b] where a=1, b=100 typically

    Returns:
        Function value f(x, y) = (a - x)^2 + b * (y - x^2)^2
    """
    a, b = params
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


def rosenbrock_torch(x, params):
    """
    Rosenbrock function for PyTorch tensors (enables automatic differentiation).

    Args:
        x: Tensor of shape (2,) containing [x, y] coordinates
        params: Tensor [a, b] where a=1, b=100 typically

    Returns:
        Function value as a tensor (gradients can be computed via .backward())
    """
    a, b = params[0], params[1]
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


def rosenbrock_for_plotting(x, y, params):
    """
    Rosenbrock function for creating 3D surface plots.

    Args:
        x, y: 2D meshgrid arrays
        params: List [a, b]

    Returns:
        2D array of function values for plotting
    """
    a, b = params
    return (a - x)**2 + b * (y - x**2)**2


# Rosenbrock parameters: a=1, b=100 (standard values)
PARAMS = [1.0, 100.0]


def rosenbrock_gradient_analytical(x, y, a=1.0, b=100.0):
    """
    Analytical gradient of the Rosenbrock function (derived by hand).

    For f(x,y) = (a - x)^2 + b * (y - x^2)^2

    The partial derivatives are:
        df/dx = -2*(a - x) + b * 2*(y - x^2) * (-2*x)
              = -2*(a - x) - 4*b*x*(y - x^2)

        df/dy = b * 2*(y - x^2)
              = 2*b*(y - x^2)

    Args:
        x, y: Coordinates at which to evaluate the gradient
        a, b: Rosenbrock parameters (default: a=1, b=100)

    Returns:
        Tuple (df/dx, df/dy) - the gradient vector
    """
    dfdx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dfdy = 2 * b * (y - x**2)
    return dfdx, dfdy


# =============================================================================
# SECTION 2: Visualize the Rosenbrock Function
# =============================================================================

print("=" * 60)
print("ROSENBROCK FUNCTION OPTIMIZATION")
print("=" * 60)

print("\n[Step 1] Creating 3D surface plot...")

# Create a grid of points for visualization
x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock_for_plotting(X, Y, PARAMS)

# Create 3D surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Rosenbrock Function: f(x,y) = (1-x)² + 100(y-x²)²')
ax.view_init(elev=30, azim=40)
fig.colorbar(surface, shrink=0.5, label='f(x,y)')
plt.tight_layout()
plt.savefig('rosenbrock_3d.png', dpi=150, bbox_inches='tight')
print("   Saved: rosenbrock_3d.png")
plt.show(block=False)


# =============================================================================
# SECTION 3: Gradient-Free Optimization (CMA-ES)
# =============================================================================

print("\n[Step 2] GRADIENT-FREE OPTIMIZATION (CMA-ES)")
print("-" * 50)

# Starting point
x0 = np.zeros(2)

# Define objective for CMA-ES
def objective_cmaes(x):
    return rosenbrock_numpy(x, PARAMS)

# Run CMA-ES optimization
start_time = time.time()
es = CMAEvolutionStrategy(x0, 0.5, {'bounds': [[-2, -2], [2, 2]], 'verbose': -1})
es.optimize(objective_cmaes, maxfun=10000)
cmaes_time = time.time() - start_time

result_cmaes = es.result
print(f"   Function evaluations: {result_cmaes.evaluations}")
print(f"   Time taken: {cmaes_time:.4f} seconds")
print(f"   Solution: x = {result_cmaes.xbest[0]:.6f}, y = {result_cmaes.xbest[1]:.6f}")
print(f"   Minimum value: {result_cmaes.fbest:.8f}")


# =============================================================================
# SECTION 4: Gradient-Based Optimization (L-BFGS with PyTorch Autograd)
# =============================================================================

print("\n[Step 3] GRADIENT-BASED OPTIMIZATION (L-BFGS with Autograd)")
print("-" * 50)

# Initialize optimization variable (requires_grad=True enables autograd)
x = torch.tensor([0.0, 0.0], device=device, requires_grad=True)
params_torch = torch.tensor(PARAMS, device=device)

# Create L-BFGS optimizer
optimizer = torch.optim.LBFGS(
    [x],
    max_iter=100,
    tolerance_grad=1e-7,
    tolerance_change=1e-9
)

# Count function evaluations
num_evals = 0

def closure():
    """Closure function required by L-BFGS optimizer."""
    global num_evals
    num_evals += 1
    optimizer.zero_grad()
    loss = rosenbrock_torch(x, params_torch)
    loss.backward()  # This computes gradients automatically!
    return loss

# Run optimization
start_time = time.time()
for i in range(100):
    loss = optimizer.step(closure)
    if x.grad is not None and torch.norm(x.grad) < 1e-7:
        break

lbfgs_time = time.time() - start_time

print(f"   Function evaluations: {num_evals}")
print(f"   Time taken: {lbfgs_time:.4f} seconds")
print(f"   Solution: x = {x[0].item():.6f}, y = {x[1].item():.6f}")
print(f"   Minimum value: {rosenbrock_torch(x, params_torch).item():.8f}")


# =============================================================================
# SECTION 5: Finite Difference Gradient Analysis
# =============================================================================

print("\n[Step 4] COMPARING ANALYTICAL vs AUTOMATIC DIFFERENTIATION")
print("-" * 50)

# Evaluation point for gradient comparison
eval_point = [0.5, 0.5]

# Method 1: Compute gradient using hand-derived ANALYTICAL formula
analytical_grad_x, analytical_grad_y = rosenbrock_gradient_analytical(
    eval_point[0], eval_point[1], PARAMS[0], PARAMS[1]
)

# Method 2: Compute gradient using PyTorch AUTOMATIC DIFFERENTIATION
x_ad = torch.tensor(eval_point, requires_grad=True)
f_ad = rosenbrock_torch(x_ad, torch.tensor(PARAMS))
f_ad.backward()
ad_grad_x = x_ad.grad[0].item()
ad_grad_y = x_ad.grad[1].item()

print(f"   Evaluation point: ({eval_point[0]}, {eval_point[1]})")
print()
print(f"   ANALYTICAL gradient (hand-derived formula):")
print(f"      df/dx = {analytical_grad_x:.10f}")
print(f"      df/dy = {analytical_grad_y:.10f}")
print()
print(f"   AUTOGRAD gradient (PyTorch automatic differentiation):")
print(f"      df/dx = {ad_grad_x:.10f}")
print(f"      df/dy = {ad_grad_y:.10f}")
print()

# Compute the difference between analytical and AD gradients
error_x = abs(analytical_grad_x - ad_grad_x)
error_y = abs(analytical_grad_y - ad_grad_y)
print(f"   DIFFERENCE (Analytical - AD):")
print(f"      |error in df/dx| = {error_x:.2e}")
print(f"      |error in df/dy| = {error_y:.2e}")
print()
print("   --> AD gives the EXACT same result as analytical differentiation!")
print("   --> No approximation error (only floating-point precision limits)")

# Use the AD gradient as the reference for FD comparison
exact_gradient = ad_grad_x


# =============================================================================
# SECTION 6: Finite Difference Gradient Analysis
# =============================================================================

print("\n[Step 5] FINITE DIFFERENCE GRADIENT ANALYSIS")
print("-" * 50)

# Forward difference approximation for various step sizes
print("\n   Computing forward difference errors...")
h_values = np.logspace(-14, -2, 500)
gradient_errors = []

for h in h_values:
    # Forward difference: df/dx ≈ [f(x+h) - f(x)] / h
    x_plus_h = [eval_point[0] + h, eval_point[1]]
    f_plus_h = rosenbrock_numpy(x_plus_h, PARAMS)
    f_current = rosenbrock_numpy(eval_point, PARAMS)

    fd_gradient = (f_plus_h - f_current) / h
    error = abs(exact_gradient - fd_gradient)
    gradient_errors.append(error)

# Plot the error vs step size
fig2 = plt.figure(figsize=(10, 6))
plt.loglog(h_values, gradient_errors, 'b-', linewidth=1.5)
plt.xlabel('Step size (h)', fontsize=12)
plt.ylabel('Gradient error |exact - FD|', fontsize=12)
plt.title('Forward Difference Gradient Error vs Step Size', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fd_error_forward.png', dpi=150, bbox_inches='tight')
print("   Saved: fd_error_forward.png")
plt.show()


# =============================================================================
# EXERCISE: Implement Central Difference Approximation
# =============================================================================
#
# TODO: Complete the central difference approximation below.
#
# Central difference formula:
#     df/dx ≈ [f(x+h) - f(x-h)] / (2h)
#
# This should be more accurate than forward difference!
# Compare the error plots to see the improvement.
#
# Uncomment and complete the code below:
# -----------------------------------------------------------------------------

# gradient_errors_central = []
#
# for h in h_values:
#     # Central difference: df/dx ≈ [f(x+h) - f(x-h)] / (2h)
#     # TODO: Compute x_plus_h and x_minus_h
#     # x_plus_h = ???
#     # x_minus_h = ???
#
#     # TODO: Evaluate function at both points
#     # f_plus_h = ???
#     # f_minus_h = ???
#
#     # TODO: Compute central difference gradient
#     # cd_gradient = ???
#
#     # TODO: Compute error
#     # error = ???
#     # gradient_errors_central.append(error)
#
# # Plot comparison
# plt.figure(figsize=(10, 6))
# plt.loglog(h_values, gradient_errors, 'b-', linewidth=1.5, label='Forward Difference')
# plt.loglog(h_values, gradient_errors_central, 'r-', linewidth=1.5, label='Central Difference')
# plt.xlabel('Step size (h)')
# plt.ylabel('Gradient error')
# plt.title('Gradient Error: Forward vs Central Difference')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('fd_error_comparison.png', dpi=150)
# plt.show()

# -----------------------------------------------------------------------------
# QUESTIONS TO CONSIDER:
#
# 1. Why do Analytical and AD gradients match exactly?
#    (Hint: AD applies the chain rule systematically, just like you would by hand)
#
# 2. Why does the FD error increase for very small h values?
#    (Hint: Think about floating-point precision and catastrophic cancellation)
#
# 3. Why does the FD error increase for very large h values?
#    (Hint: Think about the Taylor series truncation error)
#
# 4. What is the optimal step size for forward difference? For central?
#    (Hint: Forward ~ sqrt(epsilon), Central ~ cbrt(epsilon))
#
# 5. How does automatic differentiation avoid these problems entirely?
# =============================================================================


# =============================================================================
# SECTION 7: Summary and Comparison
# =============================================================================

print("\n" + "=" * 60)
print("OPTIMIZATION COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Method':<20} {'Time (s)':<12} {'Evaluations':<15} {'x*':<12} {'y*':<12}")
print("-" * 70)
print(f"{'CMA-ES':<20} {cmaes_time:<12.4f} {result_cmaes.evaluations:<15} {result_cmaes.xbest[0]:<12.6f} {result_cmaes.xbest[1]:<12.6f}")
print(f"{'L-BFGS (Autograd)':<20} {lbfgs_time:<12.4f} {num_evals:<15} {x[0].item():<12.6f} {x[1].item():<12.6f}")
print("-" * 70)
print(f"True minimum: (1.0, 1.0) with f(x,y) = 0")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. ANALYTICAL and AD gradients are IDENTICAL (AD = automated chain rule)
2. Gradient-based methods (L-BFGS) converge faster with fewer evaluations
3. Gradient-free methods (CMA-ES) are more robust but slower
4. Finite differences have a sweet spot for step size h:
   - Too small: round-off error dominates
   - Too large: truncation error dominates
5. Central difference is more accurate than forward difference
6. AD avoids all finite difference issues - exact derivatives, always!
""")

print("\nExercise completed!")
