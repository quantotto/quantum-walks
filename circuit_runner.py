"""Helper classes and functions for Quantum Galton Board simulation notebook."""

from typing import Dict, List, Tuple
from attr import dataclass

import matplotlib.pyplot as plt
from enum import Enum

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as IBMSampler
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


@dataclass
class CircuitDefinition:
    circuit_generator: callable
    coin: callable
    distribution_type: DistributionType = DistributionType.NORMAL
    title: str = ""
    kwargs: Dict = {}


def cleanup_freqs(n, shots, freqs: Dict[str, int]) -> Dict[str, float]:
    """Cleans up the frequency dictionary to ensure all bitstrings are present."""
    if not freqs:
        return {}
    new_freqs = {}
    for i in range(0, n + 1):
        bits = ["0"] * (n + 1)
        bits[i] = "1"
        bitstring = "".join(bits)
        new_freqs[bitstring] = freqs.get(bitstring, 0)
    new_freqs = dict(sorted(new_freqs.items()))
    total_counts = sum(new_freqs.values())
    if total_counts < shots:
        ratio = shots / total_counts
        for key in new_freqs:
            new_freqs[key] = new_freqs[key] * ratio
    return new_freqs


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
        self.mitigate_noise = mitigate_noise
        self.kwargs = kwargs
        self.job_runner = self._create_job_runner()
        self._freqs = []

    def _create_job_runner(self):
        """Create a job runner based on the run mode."""
        options = {"method": "matrix_product_state", "max_parallel_threads": 4}
        if self.run_mode == RunMode.NOISELESS_SIMULATOR:
            job_runner = AerSimulator(**options)
        elif self.run_mode == RunMode.NOISY_SIMULATOR:
            job_runner = AerSimulator.from_backend(self.backend, **options)
        elif self.run_mode == RunMode.REAL_DEVICE:
            job_runner = IBMSampler(mode=self.backend)
        else:
            raise ValueError(f"Unsupported run mode: {self.run_mode}")
        return job_runner

    def run_circuit(
        self, circuit: QuantumCircuit | List[QuantumCircuit]
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """Run the quantum circuit and return the frequency distribution."""
        if isinstance(circuit, QuantumCircuit):
            circuits = [circuit]
        elif isinstance(circuit, List):
            circuits = circuit
        else:
            raise ValueError(
                "Circuit must be a QuantumCircuit or a list of QuantumCircuits."
            )
        job = self.job_runner.run(circuits, shots=self.shots)
        result = job.result()
        if self.run_mode == RunMode.REAL_DEVICE:
            freqs = [res.data.distribution.get_counts() for res in result]
        else:
            if len(circuits) == 1:
                freqs = [result.get_counts()]
            else:
                freqs = result.get_counts()
        freqs_mitigated = []
        if self.mitigate_noise and self.run_mode != RunMode.NOISELESS_SIMULATOR:
            print("Mitigating noise...")
            for i, qc in enumerate(circuits):
                mit = mthree.M3Mitigation(
                    self.job_runner
                    if self.run_mode == RunMode.NOISY_SIMULATOR
                    else self.backend
                )
                mit.cals_from_system(range(qc.num_qubits))
                print("Applying correction...")
                quasi = mit.apply_correction(freqs[i], range(self.n + 1))
                probs = quasi.nearest_probability_distribution()
                probs = dict(sorted(probs.items()))
                freqs_mitigated.append(
                    {
                        bitstring: count * self.shots
                        for bitstring, count in probs.items()
                    }
                )
            print("Correction applied.")
        else:
            freqs_mitigated = [{}] * len(circuits)
        for i in range(0, self.n + 1):
            bits = ["0"] * (self.n + 1)
            bits[i] = "1"
            bitstring = "".join(bits)
            for frs in freqs:
                if bitstring not in frs:
                    frs[bitstring] = 0
            if self.mitigate_noise:
                for frs in freqs_mitigated:
                    if bitstring not in frs:
                        frs[bitstring] = 0
        self._freqs = [
            cleanup_freqs(self.n, self.shots, dict(sorted(frs.items())))
            for frs in freqs
        ]
        self._freqs_mitigated = [
            cleanup_freqs(self.n, self.shots, dict(sorted(frs.items())))
            for frs in freqs_mitigated
        ]
        return self._freqs, self._freqs_mitigated


def plot_freqs(n, freqs, title="", x_map=None, reference_values=None):
    """Plots the frequencies of bitstrings as a histogram, with optional reference line."""
    if len(freqs) != n + 1:
        print("Warning: The number of frequencies does not match the expected count.")
        return
    if x_map is not None:
        if len(x_map) != n + 1:
            print("Warning: x_map length does not match the expected count.")
            return
        x_axis = x_map
    else:
        x_axis = list(freqs.keys())
    plt.bar(x_axis, freqs.values(), label="Quantum Galton Box")
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
    freqs, freqs_mitigated = runner.run_circuit(circuit)
    if plots:
        plot_freqs(
            n,
            freqs[0],
            title=title,
            x_map=positions,
            reference_values=reference_freqs if show_reference else None,
        )
        if mitigate_noise:
            plot_freqs(
                n,
                freqs_mitigated[0],
                title=f"{title} (Mitigated)",
                x_map=positions,
                reference_values=reference_freqs if show_reference else None,
            )
    return positions, freqs[0], freqs_mitigated[0], reference_freqs


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


def run_optimized_multi_job_simulation(
    circuit_defs: List[CircuitDefinition],
    n: int,
    shots: int,
    run_mode: RunMode,
    show_reference: bool = True,
    backend: BackendV2 = None,
    mitigate_noise: bool = False,
    plots: bool = True,
) -> Tuple[List[int], Dict[str, float], List[float]]:
    """Runs the Quantum Galton Board simulation and returns the results."""
    runner = CircuitRunner(
        n, shots, run_mode, backend=backend, mitigate_noise=mitigate_noise
    )
    circuits = []
    distribution_generator = DistributionGenerator(n, shots)
    for circuit_def in circuit_defs:
        probs = distribution_generator.generate_distribution(
            circuit_def.distribution_type, generate_probs=True
        )[1]
        qc = circuit_def.circuit_generator(n, circuit_def.coin, probs=probs)
        if run_mode != RunMode.REAL_DEVICE:
            my_backend = runner.job_runner
        else:
            my_backend = backend
        qc = transpile(qc, backend=my_backend, optimization_level=1)
        print(f"Width and depth of transpiled circuit: {qc.width()}, {qc.depth()}")
        if run_mode in (RunMode.NOISY_SIMULATOR, RunMode.REAL_DEVICE):
            ai_transpiler_pass_manager = generate_ai_pass_manager(
                backend=my_backend,
                ai_optimization_level=3,
                optimization_level=3,
                ai_layout_mode="optimize",
            )
            qc = ai_transpiler_pass_manager.run(qc)
            print(f"Width and depth after AI Pass: {qc.width()}, {qc.depth()}")
        circuits.append(qc)
    freqs, freqs_mitigated = runner.run_circuit(circuits)
    positions, reference_freqs = [], []
    for i in range(len(circuit_defs)):
        pos, ref_freq = distribution_generator.generate_distribution(
            circuit_defs[i].distribution_type
        )
        positions.append(pos)
        reference_freqs.append(ref_freq)
        if plots:
            plot_freqs(
                n,
                freqs[i],
                title=circuit_defs[i].title,
                x_map=pos,
                reference_values=ref_freq if show_reference else None,
            )
            if mitigate_noise:
                plot_freqs(
                    n,
                    freqs_mitigated[i],
                    title=f"{circuit_defs[i].title} (Mitigated)",
                    x_map=pos,
                    reference_values=ref_freq if show_reference else None,
                )
    return positions, freqs, freqs_mitigated, reference_freqs
