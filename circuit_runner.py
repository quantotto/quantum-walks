"""Helper classes and functions for Quantum Galton Board simulation notebook."""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from enum import Enum

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as IBMSampler
from qiskit.providers import BackendV2
from qiskit.providers import BackendV2
from qiskit_ibm_transpiler import generate_ai_pass_manager

import mthree

from distributions import DistributionGenerator, DistributionType


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
        mitigate_noise: bool = False,
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
        self.mititate_noise = mitigate_noise
        self.kwargs = kwargs
        self.job_runner = self._create_job_runner()
        self._freqs = []

    def _create_job_runner(self):
        """Create a job runner based on the run mode."""
        if self.run_mode == RunMode.NOISELESS_SIMULATOR:
            job_runner = AerSimulator(method="automatic")
        elif self.run_mode == RunMode.NOISY_SIMULATOR:
            job_runner = AerSimulator.from_backend(self.backend, method="automatic")
        elif self.run_mode == RunMode.REAL_DEVICE:
            job_runner = IBMSampler(mode=self.backend)
        else:
            raise ValueError(f"Unsupported run mode: {self.run_mode}")
        return job_runner

    def _cleanup_freqs(self, freqs: Dict[str, int]) -> Dict[str, float]:
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
                new_freqs[key] = new_freqs[key] * ratio
        return new_freqs

    def run_circuit(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """Run the quantum circuit and return the frequency distribution."""
        job = self.job_runner.run([circuit], shots=self.shots)
        result = job.result()
        if self.run_mode == RunMode.REAL_DEVICE:
            freqs = result[0].data.distribution.get_counts()
        else:
            freqs = result.get_counts()
        if self.mititate_noise and self.run_mode != RunMode.NOISELESS_SIMULATOR:
            print("Mitigating noise...")
            mit = mthree.M3Mitigation(
                self.job_runner
                if self.run_mode == RunMode.NOISY_SIMULATOR
                else self.backend
            )
            mit.cals_from_system(range(circuit.num_qubits))
            print("Applying correction...")
            quasi = mit.apply_correction(freqs, range(self.n + 1))
            probs = quasi.nearest_probability_distribution()
            probs = dict(sorted(probs.items()))
            freqs = {
                bitstring: count * self.shots for bitstring, count in probs.items()
            }
            print("Correction applied.")
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
                color="blue",
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


def run_simulation(
    n: int,
    shots: int,
    circuit_generator: callable,
    coin: callable,
    run_mode: RunMode,
    distribution_type: DistributionType = DistributionType.NORMAL,
    title: str = "",
    show_reference: bool = True,
    backend: BackendV2 = None,
    mitigate_noise: bool = False,
    plots: bool = True,
    **kwargs,
) -> Tuple[List[int], Dict[str, float], List[float]]:
    """Runs the Quantum Galton Board simulation and returns the results."""
    runner = CircuitRunner(
        n, shots, run_mode, backend=backend, mitigate_noise=mitigate_noise
    )
    circuit = circuit_generator(n, coin, **kwargs)
    if run_mode != RunMode.REAL_DEVICE:
        my_backend = runner.job_runner
    else:
        my_backend = backend
    circuit = transpile(circuit, backend=my_backend, optimization_level=1)
    print(
        f"Width and depth of transpiled circuit: {circuit.width()}, {circuit.depth()}"
    )
    if run_mode in (RunMode.NOISY_SIMULATOR, RunMode.REAL_DEVICE):
        ai_transpiler_pass_manager = generate_ai_pass_manager(
            backend=my_backend,
            ai_optimization_level=3,
            optimization_level=3,
            ai_layout_mode="optimize",
        )
        circuit = ai_transpiler_pass_manager.run(circuit)
        print(f"Width and depth after AI Pass: {circuit.width()}, {circuit.depth()}")

    distribution_generator = DistributionGenerator(n, shots)
    positions, reference_freqs = distribution_generator.generate_distribution(
        distribution_type
    )
    freqs = runner.run_circuit(circuit)
    if plots:
        runner.plot_freqs(
            title=title,
            x_map=positions,
            reference_values=reference_freqs if show_reference else None,
        )
    return positions, freqs, reference_freqs


def ai_transpile_circuit(circuit: QuantumCircuit, backend: BackendV2) -> QuantumCircuit:
    """Transpiles the given quantum circuit for the specified backend."""
    my_circuit = transpile(circuit, backend=backend, optimization_level=1)
    ai_transpiler_pass_manager = generate_ai_pass_manager(
        backend=backend,
        ai_optimization_level=3,
        optimization_level=3,
        ai_layout_mode="optimize",
    )
    return ai_transpiler_pass_manager.run(my_circuit)
