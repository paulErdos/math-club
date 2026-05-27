#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def mobius_by_fixed_points(z, g1, g2, k):
    """Applies a Mobius transform defined by two fixed points and a multiplier k."""
    # Solving the characteristic equation for f(z):
    # (f(z) - g1) / (f(z) - g2) = k * (z - g1) / (z - g2)
    numerator = g1 * (z - g2) - k * g2 * (z - g1)
    denominator = (z - g2) - k * (z - g1)
    return numerator / denominator

def make_base_circle(center, radius):
    """Generates complex points for a circle boundary and an internal radius line."""
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = center + radius * np.exp(1j * theta)
    # Radius line pointing to the right (like a clock hand at 3 o'clock)
    line = center + np.linspace(0, radius, 50) * np.exp(1j * 0)
    return circle, line

def plot_transform_type(ax, g1, g2, k, num_iterations=25, title=""):
    ax.set_facecolor('black')
    
    # Initial circle parameters
    current_circle, current_line = make_base_circle(center=0.0 + 0.0j, radius=1.0)
    
    # Color gradient mapping (Blue -> White -> Yellow)
    colors = plt.cm.plasma(np.linspace(0, 1, num_iterations))
    
    for i in range(num_iterations):
        # Determine color based on phase of iteration to mimic the source style
        if i < num_iterations // 2:
            color = 'cyan' if title == "Hyperbolic" else 'dodgerblue'
        elif i == num_iterations // 2:
            color = 'white'
        else:
            color = 'yellow'
            
        # Specific color overrides to closely match your image breakdown
        if title == "Hyperbolic":
            color = 'dodgerblue' if i < 12 else ('white' if i == 12 else 'yellow')
        elif title == "Elliptical":
            color = 'dodgerblue' if i < 8 else ('white' if i == 8 else 'yellow')
        elif title == "Loxodromic":
            color = 'dodgerblue' if i < 18 else ('white' if i == 18 else 'yellow')

        # Plot boundary and internal radius line
        ax.plot(current_circle.real, current_circle.imag, color=color, linewidth=1)
        ax.plot(current_line.real, current_line.imag, color=color, linewidth=1)
        
        # Iterate shapes through the transform
        current_circle = mobius_by_fixed_points(current_circle, g1, g2, k)
        current_line = mobius_by_fixed_points(current_line, g1, g2, k)

    # Plot fixed points
    ax.plot(g1.real, g1.imag, 'go', markersize=6, label='Fixed Point 1' if i==0 else "")
    ax.plot(g2.real, g2.imag, 'ro', markersize=6, label='Fixed Point 2' if i==0 else "")
    
    ax.set_title(title, color='white', fontsize=14)
    ax.axis('equal')
    ax.axis('off')

# --- Execution Execution ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')

# Fixed points configuration
g1 = -1.5 + 1.5j
g2 = 1.5 - 1.5j

# 1. Hyperbolic: k is purely real (Scaling)
k_hyperbolic = 1.25 + 0.0j
plot_transform_type(axs[0], g1, g2, k_hyperbolic, num_iterations=20, title="Hyperbolic")

# 2. Elliptical: k is a pure rotation (Magnitude is 1)
theta = 0.25  # Rotation angle step
k_elliptical = np.exp(1j * theta)
plot_transform_type(axs[1], g1, g2, k_elliptical, num_iterations=16, title="Elliptical")

# 3. Loxodromic: k has both scaling and rotation components
k_loxodromic = 1.12 * np.exp(1j * 0.2)
plot_transform_type(axs[2], g1, g2, k_loxodromic, num_iterations=35, title="Loxodromic")

plt.tight_layout()
plt.show()
