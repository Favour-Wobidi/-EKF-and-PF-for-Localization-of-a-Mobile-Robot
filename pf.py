import numpy as np

from utils import minimized_angle


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        predicted_particles = np.zeros((self.num_particles, 3))  # After motion
        expected_observations = np.zeros(self.num_particles)     # z_hat
        weights = np.zeros(self.num_particles)                   # For resampling

        # For each particle, propagate motion and compute likelihood
        for i in range(self.num_particles):
            # Sample a noisy action for this particle
            noisy_u = env.sample_noisy_action(u, alphas=self.alphas).ravel()

            # Predict new particle pose
            predicted_particles[i, :] = env.forward(self.particles[i, :], noisy_u).ravel()

            # Predict observation for the given landmark from predicted pose
            expected_observations[i] = env.observe(predicted_particles[i, :], marker_id)

            # Innovation = observed - expected (normalized angle)
            innovation = minimized_angle(z - expected_observations[i])

            # Likelihood of this particle
            weights[i] = env.likelihood(innovation, self.beta)

        # Avoid division by zero or zero weights
        weights += 1.e-300
        weights /= np.sum(weights)

        # Resample particles using low-variance sampling
        self.particles, self.weights = self.resample(predicted_particles, weights)

        # Estimate mean and covariance from resampled particles
        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """
        Low-variance resampling of particles.

        Parameters:
        - particles: (n x 3) matrix of particle poses
        - weights: Corresponding normalized weights

        Returns:
        - new_particles: Resampled particle matrix
        - new_weights: Uniform weights
        """
        M = self.num_particles
        new_particles = np.zeros((M, 3))
        new_weights = np.ones(M) / M  # After resampling, all weights are equal

        # Low-variance resampling algorithm
        r = np.random.uniform(0, 1/M)
        c = weights[0]
        i = 0
        for m in range(M):
            U = r + m / M
            while U > c:
                i += 1
                c += weights[i]
            new_particles[m, :] = particles[i, :]
        return new_particles, new_weights

    def mean_and_variance(self, particles):
        """
        Compute the mean and covariance of a set of particles.

        Parameters:
        - particles: (n x 3) matrix of poses [x, y, theta]

        Returns:
        - mean: Estimated mean (3x1 vector)
        - cov: Estimated covariance (3x3 matrix)
        """
        mean = particles.mean(axis=0)

        # Circular mean for angles
        mean[2] = np.arctan2(
            np.sin(particles[:, 2]).sum(),
            np.cos(particles[:, 2]).sum()
        )

        # Center particles for covariance calculation
        zero_mean = particles - mean
        for i in range(particles.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])  # Normalize theta diff

        cov = (zero_mean.T @ zero_mean) / self.num_particles
        return mean.reshape((-1, 1)), cov