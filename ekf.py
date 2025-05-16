""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark observation."""
        
        # === Prediction Step ===
        # Predict the new mean state using the motion model
        mu_bar = env.forward(self.mu, u)              # Predicted mean (3x1)
        G = env.G(self.mu, u)                         # Jacobian of motion model w.r.t. state (3x3)
        V = env.V(self.mu, u)                         # Jacobian of motion model w.r.t. control (3x2)
        M = env.noise_from_motion(u, self.alphas)     # Control-dependent motion noise (2x2)

        # Predict the new covariance
        sigma_bar = G @ self.sigma @ G.T + V @ M @ V.T  # (3x3)

        # === Update Step ===
        # Predict the expected observation from predicted state
        z_hat = env.observe(mu_bar.ravel(), marker_id)   # Predicted observation (e.g., [bearing, range])

        # Compute the Jacobian of the observation model w.r.t. state
        H = env.H(mu_bar.ravel(), marker_id)             # (2x3)

        # Compute the innovation (difference between actual and predicted observation)
        innovation = z - z_hat                            # (2x1)
        innovation[0] = minimized_angle(innovation[0])    # Normalize angle to [-π, π]

        # Compute the innovation covariance
        S = H @ sigma_bar @ H.T + self.beta               # (2x2)

        # Compute the Kalman gain
        K = sigma_bar @ H.T @ np.linalg.inv(S)            # (3x2)

        # Update the state mean and covariance
        mu_new = mu_bar + K @ innovation                  # (3x1)
        sigma_new = sigma_bar - K @ S @ K.T               # (3x3)

        # Save the updated belief
        self.mu = mu_new
        self.sigma = sigma_new

        return self.mu, self.sigma



