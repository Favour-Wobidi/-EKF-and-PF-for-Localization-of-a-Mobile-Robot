import matplotlib.pyplot as plt
import numpy as np

# Load precomputed data from previous runs (or extract from images if needed)
# Since we only saved plots earlier, we simulate the data reading for now.
# If you want real comparison, you'd need to save the r_values and errors during `run_experiments`

# Re-define the r_values (log scale)
r_values = [1/64, 1/16, 1/4, 4, 16, 64]

# Load saved .npy data if available; otherwise use placeholders
# For demonstration, we assume mean errors were saved as .npy arrays
try:
    a_pos_error = np.load("ekf_part_a_pos_error.npy")
    b_pos_error = np.load("ekf_part_b_pos_error.npy")
    a_mahal_error = np.load("ekf_part_a_mahal_error.npy")
    b_mahal_error = np.load("ekf_part_b_mahal_error.npy")
    a_anees = np.load("ekf_part_a_anees.npy")
    b_anees = np.load("ekf_part_b_anees.npy")
except FileNotFoundError:
    print("Warning: .npy data files not found. Using placeholder dummy data.")
    a_pos_error = [0.2, 0.4, 0.8, 1.2, 1.4, 1.5]
    b_pos_error = [0.3, 0.6, 1.1, 1.8, 2.2, 2.4]

    a_mahal_error = [0.9, 1.0, 1.2, 1.3, 1.5, 1.6]
    b_mahal_error = [1.1, 1.3, 1.6, 2.0, 2.3, 2.6]

    a_anees = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    b_anees = [1.3, 1.4, 1.7, 2.0, 2.5, 2.8]

# Plot Mean Position Error Comparison
plt.figure(figsize=(8, 4))
plt.plot(r_values, a_pos_error, marker='o', label='Part A', color='blue')
plt.plot(r_values, b_pos_error, marker='s', label='Part B', color='red')
plt.xscale('log')
plt.xlabel('Noise factor r (log scale)')
plt.ylabel('Mean Position Error')
plt.title('EKF Comparison: Mean Position Error (A vs B)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ekf_comparison_mean_position_error.png")
plt.show()

# Plot Mahalanobis Error Comparison
plt.figure(figsize=(8, 4))
plt.plot(r_values, a_mahal_error, marker='^', label='Part A', color='green')
plt.plot(r_values, b_mahal_error, marker='x', label='Part B', color='orange')
plt.xscale('log')
plt.xlabel('Noise factor r (log scale)')
plt.ylabel('Mean Mahalanobis Error')
plt.title('EKF Comparison: Mahalanobis Error (A vs B)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ekf_comparison_mahalanobis_error.png")
plt.show()

# Plot ANEES Comparison
plt.figure(figsize=(8, 4))
plt.plot(r_values, a_anees, marker='D', label='Part A', color='purple')
plt.plot(r_values, b_anees, marker='v', label='Part B', color='brown')
plt.xscale('log')
plt.xlabel('Noise factor r (log scale)')
plt.ylabel('ANEES')
plt.title('EKF Comparison: ANEES (A vs B)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ekf_comparison_anees.png")
plt.show()
