# QuantumChemistrySimulations

This repo serves as a container for everything done during my master's thesis specialization project.

Data produced with this repo is found in the `data` folder, which contains `.csv` files.

In this repository you will find python scripts which contain:
- Using quantum computing to solve the electronic structure problem in quantum chemistry.
- Simulations of the Local Unitary Cluster Jastrow ansatz on a quantum circuit by utilizing the `ffsim` library and `qiskit`.
- Classical quantum circuit simulations with tensor network methods, via the `matrix_product_state` method from `qiskit-aer`.
- Sample based Quantum Diagonalization with samples from real IBM-Q jobs.
- Variational quantum eigensolver simulations on a matrix product state tensor network with various optimizers like **QN-SPSA**, **SPSA**,  **L-BFGS-B** and the **Linear Method for Optimizing Jastrow-Feenberg correlations**.
 
This repository contains scripts and usage of my development library `fqcsim` (see https://github.com/davidomanovic/fqcsim) for different experiments quantum computing applications. 

# Examples
## Born-Oppenheimer bond dissociation curves for molecules using quantum circuit ansätze (LUCJ) and methods from quantum chemistry (CC theory, FCI, HF)

### Hydrogen H2 CAS(2e,10o) in the cc-pVDZ basis (20 qubits)
<img width="750" height="837" alt="image" src="https://github.com/user-attachments/assets/517677bc-a29d-4179-abc0-35712d9b54e6" />

### Ethene C2H4 CAS(4e,4o) pi-bonding picture (8 qubits)
<img width="482" height="441" alt="image" src="https://github.com/user-attachments/assets/449c9fe2-d18a-4f5d-b2e8-7bb580856832" />

### Nitrogen N2 CAS(10e,10o) in the STO-6G basis (20 qubits)
<img width="536" height="598" alt="image" src="https://github.com/user-attachments/assets/9ac704e2-d69a-4675-bc92-287ce36f75af" />

### Tensor-Network simulations of Nitrogen N2 CAS(10e,26o) in the cc-pVDZ basis (58 qubits)
<img width="780" height="632" alt="image" src="https://github.com/user-attachments/assets/7128a077-a930-438a-a9c1-3314b161e35c" />

### Convergence study of tensor-network methods
<img width="813" height="832" alt="image" src="https://github.com/user-attachments/assets/dfb52a43-fed9-4f9e-a9f7-e2a6354bf8df" />

### Convergence of different optimizers used in VQE
<img width="981" height="623" alt="image" src="https://github.com/user-attachments/assets/ded390b0-32e6-4af8-9a9c-1ed04c88390e" />

### Numerical study on the LUCJ ansatz
<img width="792" height="726" alt="image" src="https://github.com/user-attachments/assets/0355fbd1-c2f3-4d23-8625-0837dbcbff29" />

<img width="782" height="618" alt="image" src="https://github.com/user-attachments/assets/ac0d838c-f2c0-454c-97ca-4fcbc9a5e389" />




