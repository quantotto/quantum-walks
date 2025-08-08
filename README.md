# Wiser / Womanium Project - Random Walks and Monte Carlo


## Submission Details

- Project Name: Quantum Walks and Monte Carlo
- Team Name: Quantotto
- Team Members:
  - Yevgeny Menaker (WISER enrollment ID: gst-U7OUAHJnnFkI38T)

I attest that all the work was done solely by myself (Yevgeny Menaker). The use of "we" in the presentation is merely a writing style. If external sources were used, they are attributed in References.

## Deliverables List

- Presentation deck: [Random Walks Project](random_walks_project_v4.pdf)
- Fully executed Jupyter Notebook of the solution: [qgb_qiskit.ipynb](qgb_qiskit.ipynb)
- Summary of the approach: [solution_2pager.pdf](solution_2pager.pdf)
- Python modules:
  - Circuit runner ([circuit_runner.py](circuit_runner.py)): infrastructure code to run the simulations based on provided parameters and plot the results
  - Distributions generator ([distributions.py](distributions.py)): helper module to generate samples (probabilities and frequencies) from various distributions
- Video: [available on YouTube](https://youtu.be/T_LdU5lpku8)

## Project Description

State preparation is fundamental in Quantum Computing. In this work, we are using the technique suggested by the Universal Statistical Simulator paper [1] to create variety of distributions in quantum state. The authors design a quantum version of Galton Board and demonstrate symmetric Gaussian as well as some biased distributions. Their approach results in a superposition of Hamming Weight ($HW$) 1 states. In case of the uniform distribution, they are called W states or a private case of Dicke states with $HW=1$. Those states are entangled and can be a resource in different algorithms (quantum cryptography, differential equations, optimization problems). Our solution allows any distribution to be encoded with Hamming Weight 1 states.

We will generalize the model to the $n$-level Galton Board and demonstrate Gaussian on noiseless simulator. We will then show that quantum pegs and a coin can be modified to output different distributions, such as Exponential or Hadamard Random Walk (Bi-Modal Distribution) [2].

To optimize for noisy environment, we come up with the Galton Board inspired approach to load any distribution and showcase the results on noisy simulators provided by IBM (with the `ibm_torino` noise model).

The solution is delivered as Jupyter Notebook and organized in five parts (matching the five project tasks):

- Part I (task 1) - two pages description of the chosen approach (available as PDF file [here](solution_2pager.pdf)) and also linked from the notebook.
- Part II (task 2) - Generalizing the Quantum Galton Board from the scientific paper to $n$ levels and obtaining normal (Gaussian) distribution in the tally bins.
- Part III (task 3) - Modifying the Quantum Galton Board to obtain Exponential and Hadamard Random Walk (bi-modal) Distributions and validating on noiseless simulator.
- Part IV (task 4) - Optimizing the circuits to execute on noisy simulators. We apply the technique allowing to load any distribution and show the results for Gaussian, Exponential and Hadamard distributions
- Part V (task 5) - Analyzing statistical distances of generated values against reference distributions. We also perform comprehensive study of circuits' depth depending on number of Galton Board levels

Appendix A showcases the runs of the optimized circuits on real IBM Torino device (the same backend was used for noise modeling) with and without M3 noise mitigation [3].


## References

1. "Universal Statistical Simulator", Mark Carney, Ben Varcoe (arXiv:2202.01735)
2. "Quantum random walks - an introductory overview", J. Kempe (arXiv:quant-ph/0303081)
3. "Scalable Mitigation of Measurement Errors on Quantum Computers", Paul D. Nation, Hwajung Kang, Neereja Sundaresan, and Jay M. Gambetta, PRX Quantum 2, 040326 (2021).
