#!/usr/bin/env python3
"""
=============================================================================
Exercise 3: Bézier Curves and Arc Length using Automatic Differentiation
=============================================================================

This hands-on exercise demonstrates:
  1. How to define and evaluate Bézier curves
  2. Using PyTorch autograd to compute tangent vectors (dx/dt, dy/dt)
  3. Computing arc length by integrating the tangent magnitude

A Bézier curve is defined by control points P0, P1, ..., Pn:
    B(t) = sum_{i=0}^{n} P_i * b_{i,n}(t)

where b_{i,n}(t) are Bernstein basis polynomials.

The arc length is computed as:
    L = integral from 0 to 1 of sqrt((dx/dt)^2 + (dy/dt)^2) dt

Learning objectives:
  - Understand how Bézier curves work (de Casteljau algorithm)
  - See how autograd computes derivatives without manual formulas
  - Apply differentiation to a practical CAD/geometry problem

=============================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# =============================================================================
# SECTION 1: Bézier Curve Implementation
# =============================================================================

def de_casteljau(control_points, t):
    """
    Evaluate a Bézier curve at parameter t using de Casteljau's algorithm.

    De Casteljau's algorithm recursively interpolates between control points:
    - Start with n+1 control points
    - At each level, linearly interpolate between adjacent points
    - After n levels, you get a single point on the curve

    Args:
        control_points: Tensor of shape (n+1, 2) - the control points [P0, P1, ..., Pn]
        t: Parameter value in [0, 1]

    Returns:
        Point on the curve at parameter t, shape (2,)
    """
    # Make a copy to avoid modifying the original
    points = control_points.clone()
    n = len(points) - 1

    # Recursive interpolation
    for level in range(n):
        new_points = []
        for i in range(n - level):
            # Linear interpolation: P_new = (1-t)*P_i + t*P_{i+1}
            new_point = (1.0 - t) * points[i] + t * points[i + 1]
            new_points.append(new_point)
        points = torch.stack(new_points)

    return points[0]


def evaluate_bezier_curve(control_points, num_points=100):
    """
    Evaluate a Bézier curve at many points for plotting.

    Args:
        control_points: Tensor of shape (n+1, 2)
        num_points: Number of points to evaluate

    Returns:
        Tensor of shape (num_points, 2) - points on the curve
    """
    t_values = torch.linspace(0, 1, num_points)
    curve_points = []

    for t in t_values:
        point = de_casteljau(control_points, t)
        curve_points.append(point.detach())

    return torch.stack(curve_points)


# =============================================================================
# SECTION 2: Computing Tangent Vectors using Autograd
# =============================================================================

def tangent_at_t(control_points, t):
    """
    Compute the tangent vector (dx/dt, dy/dt) at parameter t using autograd.

    This is the KEY function showing the power of automatic differentiation!
    We don't need to derive the tangent formula by hand - PyTorch does it for us.

    Args:
        control_points: Tensor of shape (n+1, 2)
        t: Parameter value (must have requires_grad=True)

    Returns:
        Tangent vector (dx/dt, dy/dt) at parameter t
    """
    # Ensure t tracks gradients
    if not t.requires_grad:
        t = t.clone().detach().requires_grad_(True)

    # Evaluate the curve at t
    point = de_casteljau(control_points, t)

    # Compute dx/dt and dy/dt using autograd
    dx_dt = torch.autograd.grad(point[0], t, create_graph=True)[0]
    dy_dt = torch.autograd.grad(point[1], t, create_graph=True)[0]

    return torch.stack([dx_dt, dy_dt])


def tangent_magnitude(control_points, t):
    """
    Compute the magnitude of the tangent vector: sqrt((dx/dt)^2 + (dy/dt)^2)

    This is the "speed" along the curve at parameter t.
    Integrating this gives the arc length.

    Args:
        control_points: Tensor of shape (n+1, 2)
        t: Parameter value

    Returns:
        Tangent magnitude (scalar)
    """
    tangent = tangent_at_t(control_points, t)
    return torch.sqrt(tangent[0]**2 + tangent[1]**2)


# =============================================================================
# SECTION 3: Arc Length Calculation
# =============================================================================

def compute_arc_length(control_points, t_start=0.0, t_end=1.0):
    """
    Compute the arc length of a Bézier curve by numerical integration.

    Arc length = integral from t_start to t_end of |dB/dt| dt
               = integral of sqrt((dx/dt)^2 + (dy/dt)^2) dt

    We use scipy.integrate.quad for numerical integration.

    Args:
        control_points: Tensor of shape (n+1, 2)
        t_start: Start parameter (default 0)
        t_end: End parameter (default 1)

    Returns:
        Arc length (scalar)
    """
    def integrand(t_val):
        t = torch.tensor(t_val, dtype=torch.float32, requires_grad=True)
        magnitude = tangent_magnitude(control_points, t)
        return magnitude.item()

    length, _ = quad(integrand, t_start, t_end, limit=100)
    return length


# =============================================================================
# SECTION 4: Main Exercise
# =============================================================================

print("=" * 60)
print("BÉZIER CURVES: Arc Length using Automatic Differentiation")
print("=" * 60)


# Define a simple quadratic Bézier curve (3 control points)
print("\n[Step 1] Defining a Quadratic Bézier Curve")
print("-" * 50)

control_points_quadratic = torch.tensor([
    [0.0, 0.0],   # P0: Start point
    [1.0, 2.0],   # P1: Control point (pulls the curve)
    [2.0, 0.0],   # P2: End point
], dtype=torch.float32)

print("   Control points:")
print("   P0 = (0, 0)  - Start")
print("   P1 = (1, 2)  - Control point")
print("   P2 = (2, 0)  - End")


# Evaluate and plot the curve
print("\n[Step 2] Plotting the Curve")
print("-" * 50)

curve_points = evaluate_bezier_curve(control_points_quadratic)

plt.figure(figsize=(10, 6))
plt.plot(curve_points[:, 0].numpy(), curve_points[:, 1].numpy(),
         'b-', linewidth=2, label='Bézier Curve')
plt.scatter(control_points_quadratic[:, 0].numpy(),
            control_points_quadratic[:, 1].numpy(),
            c='red', s=100, zorder=5, label='Control Points')
plt.plot(control_points_quadratic[:, 0].numpy(),
         control_points_quadratic[:, 1].numpy(),
         'r--', alpha=0.5, label='Control Polygon')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Bézier Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('bezier_quadratic.png', dpi=150, bbox_inches='tight')
print("   Saved: bezier_quadratic.png")
plt.show(block=False)


# Compute tangent vectors using autograd
print("\n[Step 3] Computing Tangent Vectors with Autograd")
print("-" * 50)

test_params = [0.0, 0.25, 0.5, 0.75, 1.0]
print("   t       dx/dt      dy/dt      |tangent|")
print("   " + "-" * 45)

for t_val in test_params:
    t = torch.tensor(t_val, requires_grad=True)
    tangent = tangent_at_t(control_points_quadratic, t)
    magnitude = torch.norm(tangent)
    print(f"   {t_val:.2f}    {tangent[0].item():+.4f}    {tangent[1].item():+.4f}    {magnitude.item():.4f}")


# Compute arc length
print("\n[Step 4] Computing Arc Length")
print("-" * 50)

arc_length = compute_arc_length(control_points_quadratic)
print(f"   Arc length (using autograd + integration): {arc_length:.6f}")

# For comparison: straight line distance from P0 to P2
straight_line = torch.norm(control_points_quadratic[-1] - control_points_quadratic[0])
print(f"   Straight line distance P0 to P2: {straight_line.item():.6f}")
print(f"   Curve is {(arc_length / straight_line.item() - 1) * 100:.1f}% longer than straight line")


# =============================================================================
# EXERCISE: Cubic Bézier Curve
# =============================================================================
#
# TODO: Create a CUBIC Bézier curve (4 control points) and compute its arc length.
#
# A cubic Bézier curve has 4 control points: P0, P1, P2, P3
# - P0 and P3 are the start and end points
# - P1 and P2 control the shape of the curve
#
# Uncomment and complete the code below:
# -----------------------------------------------------------------------------

# print("\n" + "=" * 60)
# print("EXERCISE: Cubic Bézier Curve")
# print("=" * 60)
#
# # TODO: Define 4 control points for a cubic Bézier curve
# control_points_cubic = torch.tensor([
#     [0.0, 0.0],   # P0: Start point
#     # TODO: Add P1
#     # TODO: Add P2
#     [3.0, 0.0],   # P3: End point
# ], dtype=torch.float32)
#
# # TODO: Evaluate and plot the curve
# # curve_points_cubic = evaluate_bezier_curve(control_points_cubic)
# # plt.figure(...)
#
# # TODO: Compute the arc length
# # arc_length_cubic = compute_arc_length(control_points_cubic)
# # print(f"Arc length of cubic curve: {arc_length_cubic:.6f}")

# -----------------------------------------------------------------------------
# QUESTIONS TO CONSIDER:
#
# 1. How does the tangent vector change along the curve?
#    (Look at dx/dt and dy/dt at different t values)
#
# 2. Where is the tangent magnitude largest? Smallest?
#    (This tells you where the curve is moving fastest/slowest)
#
# 3. How would you compute the curve's curvature using autograd?
#    (Hint: curvature involves second derivatives)
#
# 4. What happens to the arc length if you move P1 further from the line P0-P2?
# =============================================================================


# =============================================================================
# SECTION 5: Visualize Tangent Vectors
# =============================================================================

print("\n[Step 5] Visualizing Tangent Vectors")
print("-" * 50)

plt.figure(figsize=(10, 6))

# Plot the curve
plt.plot(curve_points[:, 0].numpy(), curve_points[:, 1].numpy(),
         'b-', linewidth=2, label='Bézier Curve')

# Plot tangent vectors at several points
t_samples = torch.linspace(0.1, 0.9, 5)
for t_val in t_samples:
    t = torch.tensor(t_val.item(), requires_grad=True)
    point = de_casteljau(control_points_quadratic, t)
    tangent = tangent_at_t(control_points_quadratic, t)

    # Normalize and scale for visualization
    tangent_normalized = tangent / torch.norm(tangent) * 0.3

    plt.arrow(point[0].item(), point[1].item(),
              tangent_normalized[0].item(), tangent_normalized[1].item(),
              head_width=0.05, head_length=0.03, fc='green', ec='green')

# Plot control points
plt.scatter(control_points_quadratic[:, 0].numpy(),
            control_points_quadratic[:, 1].numpy(),
            c='red', s=100, zorder=5, label='Control Points')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Bézier Curve with Tangent Vectors (computed via Autograd)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('bezier_tangents.png', dpi=150, bbox_inches='tight')
print("   Saved: bezier_tangents.png")
plt.show()


# =============================================================================
# SECTION 6: Key Takeaways
# =============================================================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. Bézier curves are defined by control points
   - The curve passes through the first and last control points
   - Interior control points "pull" the curve toward them

2. De Casteljau's algorithm evaluates points on the curve
   - Recursive linear interpolation
   - Numerically stable

3. Autograd computes tangent vectors automatically!
   - No need to derive dB/dt by hand
   - Just define the curve and call backward()

4. Arc length = integral of tangent magnitude
   - L = integral of sqrt((dx/dt)^2 + (dy/dt)^2) dt
   - Autograd provides dx/dt and dy/dt
   - Numerical integration gives the arc length

5. This same approach works for ANY differentiable curve!
   - NURBS, splines, implicit curves, etc.
   - Autograd handles the derivatives automatically
""")

print("\nExercise completed!")
