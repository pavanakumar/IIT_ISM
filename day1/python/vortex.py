#!/usr/bin/env python3
"""
=============================================================================
Exercise 4: Complex Step Differentiation - Isentropic Vortex
=============================================================================

This hands-on exercise demonstrates:
  1. Complex step differentiation - a clever trick for computing derivatives
  2. Application to fluid dynamics: computing vorticity from velocity field
  3. Comparison with analytical derivatives

Complex Step Differentiation:
  The key insight is that for an analytic function f(x):
      f(x + i*h) = f(x) + i*h*f'(x) - h²*f''(x)/2 - ...

  Taking the imaginary part:
      Im[f(x + i*h)] = h*f'(x) + O(h³)

  Therefore:
      f'(x) ≈ Im[f(x + i*h)] / h

  This avoids the cancellation errors of finite differences!
  No matter how small h is, we get accurate derivatives.

Learning objectives:
  - Understand why finite differences suffer from round-off errors
  - Learn the complex step trick for accurate derivatives
  - See an application in computational fluid dynamics (vorticity)

=============================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# =============================================================================
# SECTION 1: Isentropic Vortex Velocity Field
# =============================================================================

def velocity_perturbations(x, y, gamma, x0, y0, R):
    """
    Compute velocity perturbations for an isentropic vortex.

    The vortex velocity field is given by:
        (u', v') = (Gamma / (2*pi*R²)) * exp[(1 - (r/R)²)/2] * (y0 - y, x - x0)

    where r² = (x - x0)² + (y - y0)²

    Args:
        x, y: Coordinates (can be scalars, arrays, or complex numbers)
        gamma: Vortex strength (Gamma)
        x0, y0: Vortex center coordinates
        R: Characteristic radius

    Returns:
        (u_prime, v_prime): Velocity perturbations
    """
    dx = x - x0
    dy = y - y0
    r_squared = dx**2 + dy**2
    R_squared = R**2

    # Exponential term
    exp_term = np.exp(-r_squared / R_squared / 2)

    # Velocity perturbations
    factor = gamma / (2 * np.pi * R_squared)
    u_prime = -factor * dy * exp_term
    v_prime = factor * dx * exp_term

    return u_prime, v_prime


# =============================================================================
# SECTION 2: Complex Step Differentiation
# =============================================================================

def vorticity_complex_step(x, y, gamma, x0, y0, R, epsilon=1e-30):
    """
    Compute vorticity using COMPLEX STEP differentiation.

    Vorticity is defined as:
        omega = dv/dx - du/dy

    Using complex step:
        dv/dx ≈ Im[v(x + i*eps, y)] / eps
        du/dy ≈ Im[u(x, y + i*eps)] / eps

    The beauty of complex step: we can use TINY epsilon (1e-30!)
    without any round-off errors.

    Args:
        x, y: Coordinates (real arrays)
        gamma: Vortex strength
        x0, y0: Vortex center
        R: Characteristic radius
        epsilon: Complex step size (can be very small!)

    Returns:
        Vorticity field
    """
    # Make complex versions of coordinates
    x_complex = x.astype(complex)
    y_complex = y.astype(complex)

    # Compute dv/dx using complex step in x direction
    x_perturbed = x_complex + 1j * epsilon
    _, v_at_x_plus_ih = velocity_perturbations(x_perturbed, y_complex, gamma, x0, y0, R)
    dv_dx = np.imag(v_at_x_plus_ih) / epsilon

    # Compute du/dy using complex step in y direction
    y_perturbed = y_complex + 1j * epsilon
    u_at_y_plus_ih, _ = velocity_perturbations(x_complex, y_perturbed, gamma, x0, y0, R)
    du_dy = np.imag(u_at_y_plus_ih) / epsilon

    # Vorticity = dv/dx - du/dy
    vorticity = dv_dx - du_dy

    return np.real(vorticity)


def vorticity_finite_difference(x, y, gamma, x0, y0, R, h=1e-6):
    """
    Compute vorticity using FINITE DIFFERENCE (central difference).

    dv/dx ≈ [v(x+h) - v(x-h)] / (2h)
    du/dy ≈ [u(y+h) - u(y-h)] / (2h)

    Args:
        x, y: Coordinates
        gamma: Vortex strength
        x0, y0: Vortex center
        R: Characteristic radius
        h: Step size

    Returns:
        Vorticity field
    """
    # Compute dv/dx using central difference
    _, v_plus = velocity_perturbations(x + h, y, gamma, x0, y0, R)
    _, v_minus = velocity_perturbations(x - h, y, gamma, x0, y0, R)
    dv_dx = (v_plus - v_minus) / (2 * h)

    # Compute du/dy using central difference
    u_plus, _ = velocity_perturbations(x, y + h, gamma, x0, y0, R)
    u_minus, _ = velocity_perturbations(x, y - h, gamma, x0, y0, R)
    du_dy = (u_plus - u_minus) / (2 * h)

    # Vorticity = dv/dx - du/dy
    vorticity = dv_dx - du_dy

    return vorticity


# =============================================================================
# SECTION 3: Main Exercise
# =============================================================================

print("=" * 60)
print("COMPLEX STEP DIFFERENTIATION: Isentropic Vortex")
print("=" * 60)

# Vortex parameters
gamma = 1.0       # Vortex strength
x0, y0 = 0.0, 0.0 # Vortex center
R = 0.1           # Characteristic radius
grid_size = 100   # Grid resolution

print(f"\nVortex Parameters:")
print(f"   Strength (Gamma): {gamma}")
print(f"   Center: ({x0}, {y0})")
print(f"   Radius: {R}")


# Create computational grid
print("\n[Step 1] Creating computational grid...")
x_1d = np.linspace(-0.5, 0.5, grid_size)
y_1d = np.linspace(-0.5, 0.5, grid_size)
X, Y = np.meshgrid(x_1d, y_1d)


# Compute velocity field
print("\n[Step 2] Computing velocity field...")
U, V = velocity_perturbations(X, Y, gamma, x0, y0, R)
velocity_magnitude = np.sqrt(U**2 + V**2)

# Plot velocity magnitude
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, velocity_magnitude, levels=20, cmap='viridis')
plt.colorbar(label='Velocity Magnitude')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Isentropic Vortex: Velocity Magnitude')
plt.axis('equal')
plt.tight_layout()
plt.savefig('vortex_velocity.png', dpi=150, bbox_inches='tight')
print("   Saved: vortex_velocity.png")
plt.show(block=False)


# Compute vorticity using complex step
print("\n[Step 3] Computing vorticity using COMPLEX STEP...")
vorticity_cs = vorticity_complex_step(X, Y, gamma, x0, y0, R, epsilon=1e-30)

# Plot vorticity
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, vorticity_cs, levels=20, cmap='RdBu_r')
plt.colorbar(label='Vorticity')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vorticity (Complex Step Method, eps=1e-30)')
plt.axis('equal')
plt.tight_layout()
plt.savefig('vortex_vorticity.png', dpi=150, bbox_inches='tight')
print("   Saved: vortex_vorticity.png")
plt.show(block=False)


# =============================================================================
# SECTION 4: Compare Complex Step vs Finite Difference
# =============================================================================

print("\n[Step 4] Comparing Complex Step vs Finite Difference")
print("-" * 50)

# Test point for comparison
test_x, test_y = 0.05, 0.05

# Complex step with various epsilon values
print("\n   Complex Step Method (varying epsilon):")
print("   epsilon          vorticity")
print("   " + "-" * 35)
for eps in [1e-6, 1e-10, 1e-20, 1e-30, 1e-50, 1e-100]:
    vort = vorticity_complex_step(
        np.array([test_x]), np.array([test_y]),
        gamma, x0, y0, R, epsilon=eps
    )[0]
    print(f"   {eps:<15.0e}  {vort:.10f}")

# Finite difference with various step sizes
print("\n   Finite Difference Method (varying h):")
print("   h                vorticity")
print("   " + "-" * 35)
for h in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    vort = vorticity_finite_difference(
        np.array([test_x]), np.array([test_y]),
        gamma, x0, y0, R, h=h
    )[0]
    print(f"   {h:<15.0e}  {vort:.10f}")


# =============================================================================
# EXERCISE: Implement the Analytical Vorticity Formula
# =============================================================================
#
# The vorticity can be derived analytically from the velocity equations.
#
# For the isentropic vortex:
#     omega = (Gamma / (2*pi*R²)) * [2 - (r/R)²] * exp[(1 - (r/R)²)/2]
#
# TODO: Implement this formula and compare with the complex step result.
#
# Uncomment and complete the code below:
# -----------------------------------------------------------------------------

# def vorticity_analytical(x, y, gamma, x0, y0, R):
#     """
#     Compute vorticity using the ANALYTICAL formula (derived by hand).
#
#     omega = (Gamma / (2*pi*R²)) * [2 - (r/R)²] * exp[(1 - (r/R)²)/2]
#
#     Args:
#         x, y: Coordinates
#         gamma: Vortex strength
#         x0, y0: Vortex center
#         R: Characteristic radius
#
#     Returns:
#         Vorticity field
#     """
#     dx = x - x0
#     dy = y - y0
#     r_squared = dx**2 + dy**2
#     R_squared = R**2
#
#     # TODO: Implement the analytical formula
#     # term1 = ???
#     # term2 = ???
#     # term3 = ???
#     # vorticity = term1 * term2 * term3
#
#     return vorticity
#
# # Compare analytical with complex step
# vorticity_analytical_result = vorticity_analytical(X, Y, gamma, x0, y0, R)
# error = np.abs(vorticity_cs - vorticity_analytical_result)
# print(f"\nMax error (Complex Step vs Analytical): {np.max(error):.2e}")

# -----------------------------------------------------------------------------
# QUESTIONS TO CONSIDER:
#
# 1. Why can complex step use epsilon = 1e-100 but finite difference cannot?
#    (Hint: What happens when you subtract two nearly equal numbers?)
#
# 2. At what h does finite difference start losing accuracy? Why?
#
# 3. Does complex step have any disadvantages compared to finite difference?
#    (Hint: Think about code complexity and applicability)
#
# 4. How does complex step compare to automatic differentiation?
# =============================================================================


# =============================================================================
# SECTION 5: Key Takeaways
# =============================================================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. Complex Step Differentiation:
   - f'(x) ≈ Im[f(x + i*h)] / h
   - Works for ANY analytic function
   - Can use TINY h (1e-100!) without round-off errors
   - No cancellation because we're taking imaginary part, not subtracting

2. Finite Difference Problems:
   - f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
   - Subtracting nearly equal numbers causes cancellation errors
   - Too small h → round-off dominates
   - Too large h → truncation dominates
   - Sweet spot around h ≈ 1e-8 for central difference

3. Complex Step Advantages:
   - Machine-precision accuracy
   - Simple to implement (if function supports complex numbers)
   - No trade-off between truncation and round-off

4. Complex Step Limitations:
   - Function must be analytic (no branches, abs, etc.)
   - Requires complex arithmetic support
   - Slightly slower than finite difference

5. In Practice:
   - Use automatic differentiation when available (best option!)
   - Use complex step for validation or simple functions
   - Use finite difference as last resort
""")

print("\nExercise completed!")
