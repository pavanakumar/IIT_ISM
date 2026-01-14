#!/usr/bin/env python3
"""
Ackley Function Optimization: Gradient vs Gradient-Free Methods

This script demonstrates:
1. 3D visualization of the ackley function
2. Gradient-free optimization (CMAES)
3. L-BFGS-B finite-difference gradient optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.optimize import minimize
from cma import CMAEvolutionStrategy
import warnings
warnings.filterwarnings('ignore')

# Define the Ackley function
def ackley_numpy(x, p):
    """Ackley function for numpy arrays (for scipy optimization)"""
    return -p[0] * np.exp(-p[1] * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(p[2] * x))) + p[0] + np.exp(1)

def ackley_for_plot(x, y, p):
    """Ackley function for plotting (numpy arrays)"""
    # Stack x and y coordinates and apply function element-wise
    result = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i, j] = ackley_numpy(np.array([x[i, j], y[i, j]]), p)
    return result

# Parameters for ackley function: [a, b, c] = [20, 0.2, 2π]
param = np.array([20.0, 0.2, 2*np.pi])
# Test from more challenging starting points
challenging_starts = np.array([
    [3.0, 3.0]
])
# Initial guess for search
x0 = challenging_starts[0]
# Evaluation point
eval_point = np.array([0.5, 0.5])

print("="*60)
print("ACKLEY FUNCTION OPTIMIZATION COMPARISON")
print("="*60)

# 1. Create 3D surface plot
print("\n1. Creating 3D surface plot...")
x_range = np.linspace(-4, 4, 50)  # Reduced resolution for faster computation
y_range = np.linspace(-4, 4, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = ackley_for_plot(X, Y, param)

fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('f(x1, x2)')
ax1.set_title('Ackley Function')
ax1.view_init(elev=30, azim=40)
fig1.colorbar(surf)
plt.tight_layout()
plt.savefig('ackley_3d_plot.png', dpi=150, bbox_inches='tight')
print("3D plot saved as ackley_3d_plot.png")

# Gradient-free optimization using CMA-ES
print("\n2. OPTIMIZATION USING GRADIENT-FREE METHOD (CMA-ES)")
print("-" * 50)

def objective_cmaes(x):
    """Objective function for CMA-ES"""
    return ackley_numpy(x, param)

# CMA-ES optimization
start_time = time.time()
es = CMAEvolutionStrategy(x0, 2.0, {'bounds': [[-4, -4], [4, 4]], 'verbose': -1})
es.optimize(objective_cmaes, maxfun=5000)
cmaes_time = time.time() - start_time

cmaes_result = es.result
print(f"Total number of function evaluations: {cmaes_result.evaluations}")
print(f"Time taken: {cmaes_time:.6f} s")
print(f"x[1] = {cmaes_result.xbest[0]:.6f}")
print(f"x[2] = {cmaes_result.xbest[1]:.6f}")
print(f"Final function value: {cmaes_result.fbest:.8f}")

# Scipy L-BFGS-B optimization for comparison
print("\n3. OPTIMIZATION USING SCIPY L-BFGS-B")
print("-" * 50)

start_time = time.time()

# Numerical gradient estimation
def objective_scipy(x):
    return ackley_numpy(x, param)

result_scipy = minimize(objective_scipy, x0, method='L-BFGS-B',
                       options={'gtol': 1e-6, 'ftol': 1e-9})
scipy_time = time.time() - start_time

print(f"Total number of function evaluations: {result_scipy.nfev}")
print(f"Time taken: {scipy_time:.6f} s")
print(f"x[1] = {result_scipy.x[0]:.6f}")
print(f"x[2] = {result_scipy.x[1]:.6f}")
print(f"Final function value: {result_scipy.fun:.8f}")
print(f"Optimization successful: {result_scipy.success}")

print("\n4. OPTIMIZATION COMPARISON SUMMARY")
print("-" * 50)
print(f"{'Method':<20} {'Time (s)':<12} {'Function Evals':<15} {'Final x1':<12} {'Final x2':<12}")
print("-" * 70)
print(f"{'CMA-ES':<20} {cmaes_time:<12.6f} {cmaes_result.evaluations:<15} {cmaes_result.xbest[0]:<12.6f} {cmaes_result.xbest[1]:<12.6f}")
print(f"{'Scipy L-BFGS-B':<20} {scipy_time:<12.6f} {result_scipy.nfev:<15} {result_scipy.x[0]:<12.6f} {result_scipy.x[1]:<12.6f}")

print(f"\nParameter values: a={param[0]}, b={param[1]:.1f}, c={param[2]:.4f}")
print(f"Global minimum should be at (0, 0) with value ≈ 0")
print(f"Test: f([0, 0]) = {ackley_numpy(np.array([0, 0]), param):.8f}")
