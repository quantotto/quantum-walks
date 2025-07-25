"""Helper classes and functions for Quantum Galton Board simulation notebook."""

from typing import Dict

import matplotlib.pyplot as plt
from enum import Enum

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as IBMSampler
from qiskit.providers import BackendV2


class RunMode(Enum):
    """Enumeration for different run modes of the Quantum Galton Board simulation."""

    NOISELESS_SIMULATOR = "noiseless_simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    REAL_DEVICE = "real_device"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, mode_str: str):
        """Convert a string to a RunMode enum."""
        return cls(mode_str.lower())


class CircuitRunner:
    def __init__(
        self,
        n: int,
        shots: int,
        run_mode: RunMode,
        backend: BackendV2 = None,
        **kwargs,
    ):
        self.n = n
        self.shots = shots
        if run_mode != RunMode.NOISELESS_SIMULATOR and backend is None:
            raise ValueError(
                "Backend must be specified for noisy or non-simulator runs."
            )
        self.run_mode = run_mode
        self.backend = backend
        self.kwargs = kwargs
        self.job_runner = self._create_job_runner()
        self._freqs = []

    def _create_job_runner(self):
        """Create a job runner based on the run mode."""
        if self.run_mode == RunMode.NOISELESS_SIMULATOR:
            job_runner = AerSimulator()
        elif self.run_mode == RunMode.NOISY_SIMULATOR:
            job_runner = AerSimulator.from_backend(self.backend)
        elif self.run_mode == RunMode.REAL_DEVICE:
            job_runner = IBMSampler(mode=self.backend)
        else:
            raise ValueError(f"Unsupported run mode: {self.run_mode}")
        return job_runner

    def _cleanup_freqs(self, freqs: Dict[str, int]) -> Dict[str, int]:
        """Cleans up the frequency dictionary to ensure all bitstrings are present."""
        new_freqs = {}
        for i in range(0, self.n + 1):
            bits = ["0"] * (self.n + 1)
            bits[i] = "1"
            bitstring = "".join(bits)
            new_freqs[bitstring] = freqs.get(bitstring, 0)
        new_freqs = dict(sorted(new_freqs.items()))
        total_counts = sum(new_freqs.values())
        if total_counts < self.shots:
            ratio = self.shots / total_counts
            for key in new_freqs:
                new_freqs[key] = int(new_freqs[key] * ratio)
        return new_freqs

    def run_circuit(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """Run the quantum circuit and return the frequency distribution."""
        job = self.job_runner.run([circuit], shots=self.shots)
        result = job.result()
        if self.run_mode == RunMode.REAL_DEVICE:
            freqs = result[0].data.distribution.get_counts()
        else:
            freqs = result.get_counts()
        for i in range(0, self.n + 1):
            bits = ["0"] * (self.n + 1)
            bits[i] = "1"
            bitstring = "".join(bits)
            if bitstring not in freqs:
                freqs[bitstring] = 0
        self._freqs = self._cleanup_freqs(dict(sorted(freqs.items())))
        return self._freqs

    def plot_freqs(self, title="", x_map=None, reference_values=None):
        """Plots the frequencies of bitstrings as a histogram, with optional reference line."""
        if len(self._freqs) != self.n + 1:
            print(
                "Warning: The number of frequencies does not match the expected count."
            )
            return
        if x_map is not None:
            if len(x_map) != self.n + 1:
                print("Warning: x_map length does not match the expected count.")
                return
            x_axis = x_map
        else:
            x_axis = list(self._freqs.keys())
        plt.bar(x_axis, self._freqs.values(), label="Quantum Galton Box")
        if reference_values is not None:
            plt.plot(
                x_axis,
                reference_values,
                color="red",
                marker="o",
                linestyle="-",
                label="Reference",
            )
        plt.xlabel("Position")
        plt.ylabel("Frequency")
        plt.title(title)
        if reference_values is not None:
            plt.legend()
        plt.show()
