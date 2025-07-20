# Generating Statistical Distribution with Quantum Galton Board

State preparation is fundamental in Quantum Computing, Different Distributions represented by multi-qubits quantum states can serve as inputs to various quantum algorithms whether we are solving differential equations or optimization problems.

In this work, we are using the technique suggested by the Universal Statistical Simulator paper to create variety of distributions in quantum state. The authors design a quantum version of Galton Board and demonstrate symmetric Gaussian as well as some biased distributions.

We will generalize this model to n-level Galton Board and demonstrate Gaussianon noiseless simulator. We will then show that quantum pegs and a coin can be modified to output different distributions, such as Exponential or Hadamard Random Walk.

## Our approach

Galton Board can be viewed as a decision tree (though technically it is a Directed Acyclic Graph), where each node (peg) is associated with probabilities to go left or right. One can also say that a node is acting as a random coin deciding what the next step should be.