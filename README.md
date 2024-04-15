# Quantum K-Nearest Neighbors: Utilizing QRAM and SWAP-Test Techniques for Enhanced Performance
This work presents an algorithm for a quantum K-Nearest Neighbor (K-NN) classifier.
The algorithm uses angle encoding through a Quantum Random Access Memory (QRAM) that achieves $\mathrm{O}(\log (\mathrm{n}))$ space complexity. 
It employs Grover's algorithm, combined with the quantum SWAP-Test, to identify similar states and determine the nearest neighbors with high probability with $\mathrm{O}\left(\sqrt{n}\right)$ search complexity. We implement a simulation of the algorithm using IBM's Qiskit, the Iris dataset, and different angle encodings. 
Finally, we compare the proposed approach results with the equivalent classical K-NN, yielding an accuracy rate of 97.8\%.
We test multiple QRAM cell sizes (8, 16, 32, 64, 128), with ten trials per size, finding accuracy values ranging from 99.3 $\pm 1.0$\% to 93.9 $\pm 3.4$\%.


# Authors

Alberto Maldonado-Romo, Jesus Yalja Montiel Perez, Victor Onofre, Javier Maldonado-Romo, Juan Humberto Sossa Azuela, and Gabriel N. Perdue


