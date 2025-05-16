""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import minimized_angle


class Field:
    NUM_MARKERS = 6

    INNER_OFFSET_X = 32
    INNER_OFFSET_Y = 13

    INNER_SIZE_X = 420
    INNER_SIZE_Y = 270

    COMPLETE_SIZE_X = INNER_SIZE_X + 2 * INNER_OFFSET_X
    COMPLETE_SIZE_Y = INNER_SIZE_Y + 2 * INNER_OFFSET_Y

    MARKER_OFFSET_X = 21
    MARKER_OFFSET_Y = 0

    MARKER_DIST_X = 442
    MARKER_DIST_Y = 292

    MARKERS = (1, 2, 3, 4, 5, 6)

    MARKER_X_POS = {
        1: MARKER_OFFSET_X,
        2: MARKER_OFFSET_X + 0.5 * MARKER_DIST_X,
        3: MARKER_OFFSET_X + MARKER_DIST_X,
        4: MARKER_OFFSET_X + MARKER_DIST_X,
        5: MARKER_OFFSET_X + 0.5 * MARKER_DIST_X,
        6: MARKER_OFFSET_X,
    }

    MARKER_Y_POS = {
        1: MARKER_OFFSET_Y,
        2: MARKER_OFFSET_Y,
        3: MARKER_OFFSET_Y,
        4: MARKER_OFFSET_Y + MARKER_DIST_Y,
        5: MARKER_OFFSET_Y + MARKER_DIST_Y,
        6: MARKER_OFFSET_Y + MARKER_DIST_Y,
    }


    def __init__(self, alphas, beta):
        self.alphas = alphas
        self.beta = beta

    # def G(self, x, u):
    #     """Compute the Jacobian of the dynamics with respect to the state."""
    #     prev_x, prev_y, prev_theta = x.ravel()
    #     rot1, trans, rot2 = u.ravel()
    #     g= np.array([[1,0,-trans(np.sin(prev_theta+rot1))],[0,1,trans(np.cos(prev_theta+rot1))],[0,0,1]])
    #     return g
    # # YOUR IMPLEMENTATION HERE

    # def V(self, x, u):
    #     """Compute the Jacobian of the dynamics with respect to the control."""
    #     prev_x, prev_y, prev_theta = x.ravel()
    #     rot1, trans, rot2 = u.ravel()
    #     v= np.array([[-trans(np.sin(prev_theta+rot1)),np.cos(prev_theta+rot1),0],[trans(np.cos(prev_theta+rot1)),np.sin(prev_theta+rot1),0],[1,0,1]])
    #     return v
    # # YOUR IMPLEMENTATION HERE

    # def H(self, x, marker_id):
    #     """Compute the Jacobian of the observation with respect to the state."""
    #     prev_x, prev_y, prev_theta = x.ravel()
    #     x_v=self.MARKER_X_POS(marker_id) - prev_x
    #     y_v=self.MARKER_Y_POS(marker_id) -prev_y
    #     q=(x_v)*2+(y_v)*2
    #     H=np.array([[y_v/q,-x_v/q,-1]])
    #     return H
    # # YOUR IMPLEMENTATION HERE

    def G(self, x, u):
        # Jacobian of motion model w.r.t. state x
        _, _, theta = x.ravel()
        rot1, trans, _ = u.ravel()
        theta += rot1
        
        return np.array([
            [1, 0, -trans * np.sin(theta)],
            [0, 1,  trans * np.cos(theta)],
            [0, 0, 1]
        ])

    def V(self, x, u):
        # Jacobian of motion model w.r.t. control u
        _, _, theta = x.ravel()
        rot1, trans, _ = u.ravel()
        theta += rot1

        return np.array([
            [-trans * np.sin(theta), np.cos(theta), 0],
            [ trans * np.cos(theta), np.sin(theta), 0],
            [1, 0, 1]
        ])

    def H(self, x, marker_id):
        # Jacobian of observation model w.r.t. state x
        x_r, y_r, theta = x.ravel()
        x_m = self.MARKER_X_POS[marker_id]
        y_m = self.MARKER_Y_POS[marker_id]
        dx = x_m - x_r
        dy = y_m - y_r
        q = dx**2 + dy**2

        # Derivative of bearing w.r.t. state [x, y, theta]
        return np.array([[-dy/q, dx/q, -1]])

    def forward(self, x, u):
        # Computes next state given current state and control (motion model)
        x_prev, y_prev, theta_prev = x
        rot1, trans, rot2 = u

        theta = theta_prev + rot1
        x_new = x_prev + trans * np.cos(theta)
        y_new = y_prev + trans * np.sin(theta)
        theta_new = minimized_angle(theta + rot2)

        return np.array([x_new, y_new, theta_new]).reshape((-1, 1))

    def get_marker_id(self, step):
        return ((step // 2) % self.NUM_MARKERS) + 1

    def observe(self, x, marker_id):
        dx = self.MARKER_X_POS[marker_id] - x[0]
        dy = self.MARKER_Y_POS[marker_id] - x[1]
        bearing = np.arctan2(dy, dx) - x[2]
        return np.array([minimized_angle(bearing)]).reshape((-1, 1))

    def noise_from_motion(self, u, alphas):
        # Computes motion noise covariance matrix
        rot1, trans, rot2 = u.ravel()
        var_rot1 = alphas[0]*rot1**2 + alphas[1]*trans**2
        var_trans = alphas[2]*trans**2 + alphas[3]*(rot1**2 + rot2**2)
        var_rot2 = alphas[0]*rot2**2 + alphas[1]*trans**2

        return np.diag([var_rot1, var_trans, var_rot2])

    def likelihood(self, innovation, beta):
        # Gaussian likelihood computation for observation innovation
        norm = np.sqrt(np.linalg.det(2 * np.pi * beta))
        inv_beta = np.linalg.inv(beta)
        return np.exp(-0.5 * innovation.T.dot(inv_beta).dot(innovation)) / norm

    def sample_noisy_action(self, u, alphas=None):
        if alphas is None:
            alphas = self.alphas
        cov = self.noise_from_motion(u, alphas)
        return np.random.multivariate_normal(u.ravel(), cov).reshape((-1, 1))

    def sample_noisy_observation(self, x, marker_id, beta=None):
        if beta is None:
            beta = self.beta
        z = self.observe(x, marker_id)
        return np.random.multivariate_normal(z.ravel(), beta).reshape((-1, 1))

    def get_figure(self):
        return plt.figure(1)

    def rollout(self, x0, policy, num_steps, dt=0.1):
        states_noisefree = np.zeros((num_steps, 3))
        states_real = np.zeros((num_steps, 3))
        action_noisefree = np.zeros((num_steps, 3))
        obs_noisefree = np.zeros((num_steps, 1))
        obs_real = np.zeros((num_steps, 1))

        x_noisefree = x_real = x0

        for i in range(num_steps):
            t = i * dt
            u_noisefree = policy(x_real, t)
            x_noisefree = self.forward(x_noisefree, u_noisefree)

            u_real = self.sample_noisy_action(u_noisefree)
            x_real = self.forward(x_real, u_real)

            marker_id = self.get_marker_id(i)
            z_noisefree = self.observe(x_real, marker_id)
            z_real = self.sample_noisy_observation(x_real, marker_id)

            states_noisefree[i, :] = x_noisefree.ravel()
            states_real[i, :] = x_real.ravel()
            action_noisefree[i, :] = u_noisefree.ravel()
            obs_noisefree[i, :] = z_noisefree.ravel()
            obs_real[i, :] = z_real.ravel()

        states_noisefree = np.concatenate([x0.T, states_noisefree], axis=0)
        states_real = np.concatenate([x0.T, states_real], axis=0)

        return (
            states_noisefree, states_real,
            action_noisefree,
            obs_noisefree, obs_real
        )
