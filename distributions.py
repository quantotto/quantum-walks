"""Helper module to generate reference distributions
for comparisons in Quantum Galton Board simulations.
"""

from typing import List, Tuple
import numpy as np
from enum import Enum


class DistributionType(Enum):
    """Enumeration for different types of distributions used in Quantum Galton Board simulations."""

    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    HADAMARD = "hadamard"
    NORMAL = "normal"


class DistributionGenerator:
    """Generates reference distributions for Quantum Galton Board simulations.
    For each distribution type, it returns a tuple of two lists of integers:
    - the first list indicates a position on the Galton Board
    - the second list indicates the number of particles at each position proportional to the probability
    """

    def __init__(self, n: int, shots: int = 2048):
        self.n = n
        self.shots = shots

    def uniform(self, generate_probs: bool = False) -> Tuple[List[int], List[float]]:
        """Generates a uniform distribution for the Galton Board."""
        uniform = np.array([1 / (self.n + 1)] * (self.n + 1))
        if not generate_probs:
            uniform *= self.shots
        return np.arange(0, self.n + 1).tolist(), uniform.tolist()

    def exponential(
        self, lambda_rate: float = 1, generate_probs: bool = False
    ) -> Tuple[List[int], List[float]]:
        """Generates an exponential distribution for the Galton Board."""
        x = np.arange(0, self.n + 1)
        exponential = np.exp(-lambda_rate * x)
        exponential /= exponential.sum()
        if not generate_probs:
            exponential = exponential * self.shots
        return x.tolist(), exponential.tolist()

    def hadamard(self, generate_probs: bool = False) -> Tuple[List[int], List[float]]:
        """Generates a Hadamard distribution for the Galton Board."""
        hadamard_op = lambda: np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        position_range = 2 * self.n + 1
        mid = position_range // 2
        state = np.zeros((2, position_range), dtype=complex)
        state[:, mid] = np.array([1, 1j])
        H = hadamard_op()
        for _ in range(self.n):
            coin_transformed = np.zeros_like(state, dtype=complex)
            for pos in range(position_range):
                coin_state = state[:, pos]
                coin_transformed[:, pos] = H @ coin_state
            new_state = np.zeros_like(state, dtype=complex)
            for pos in range(position_range):
                for coin in [0, 1]:
                    shift = -1 if coin == 0 else 1
                    new_pos = pos + shift
                    if 0 <= new_pos < position_range:
                        new_state[coin, new_pos] += coin_transformed[coin, pos]
            state = new_state
        total_prob = np.sum(np.abs(state) ** 2)
        hadamard_dist = np.sum(np.abs(state) ** 2, axis=0) / total_prob
        if not generate_probs:
            hadamard_dist = hadamard_dist * self.shots
        positions = np.arange(-self.n, self.n + 1)
        return positions[::2].tolist(), hadamard_dist[::2].tolist()

    def normal(self, generate_probs: bool = False) -> Tuple[List[int], List[float]]:
        """Generates a normal distribution approximation for the Galton Board.
        if gernerate_probs is True, returns probabilities instead of frequencies.
        """
        left = -(self.n / 2)
        right = left + self.n + 1
        positions = np.arange(left, right)
        mean = 0
        std = np.sqrt(self.n)
        gaussian = np.exp(-0.5 * ((2 * positions - mean) / std) ** 2)
        gaussian /= gaussian.sum()
        if not generate_probs:
            gaussian *= self.shots
        return positions.tolist(), gaussian.tolist()

    def generate_distribution(
        self,
        distribution_type: DistributionType,
        generate_probs: bool = False,
        **kwargs,
    ) -> Tuple[List[int], List[float]]:
        """Generates a distribution based on the specified type."""
        distribution_generators = {
            DistributionType.UNIFORM: self.uniform,
            DistributionType.EXPONENTIAL: self.exponential,
            DistributionType.HADAMARD: self.hadamard,
            DistributionType.NORMAL: self.normal,
        }
        if distribution_type not in distribution_generators:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")
        return distribution_generators[distribution_type](
            generate_probs=generate_probs, **kwargs
        )
