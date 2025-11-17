# Simulation Framework

## DeGroot Model from [1]
The belief dynamics follow the extended DeGroot-type update:

\[
x_i(t+1) = \sum_{j=1}^{n} W_{ij} \, C_i \, x_j(t)
\]

where  
- \( W \) is the social influence (row-stochastic) matrix,  
- \( C_i \) is the logic matrix of agent \( i \),  
- \( x_i(t) \) denotes the belief vector of agent \( i \) at time \( t \).

---

# Pipeline Overview
1. Initialize \( W \), \( C \), and \( x(0) \)  
2. Simulate belief evolution over time  
3. Visualize trajectories and compute relevant metrics  

---

# Generation of Social Influence Matrix \( W \)
**Purpose:** Evaluate how different social network structures affect the final steady-state beliefs.

### Random Network Models
- **Watts–Strogatz (WS)** small-world model  
- **Barabási–Albert (BA)** scale-free model  
- **Erdős–Rényi (ER)** random graph model  
- **Random-Regular (RR)** model  

### Design Requirements
- Support flexible population sizes  
- \( W \) must be **row-stochastic**  
- Allow optional randomness in model parameters  
- Ensure stability (spectral radius \( \rho(W) < 1 \) if required)

---

# Generation of Logic Matrices \( C_i \)
- Structure motivated by the research problem  
- Typically lower-triangular, capturing topic dependencies  
- Absolute row-stochastic normalization  
- Optional heterogeneity across agents  
- Optional Beta-distributed coefficients  

---

# Dynamic Simulation
- Iterative update over time steps  
- Optional **early stopping** when convergence is reached  
- Time horizon selected according to experimental objectives  

---

# Visualization
- Define appropriate metrics for comparison  
- Select meaningful dimensions (e.g., per-topic trajectories, agent clusters)  
- Provide clear and persuasive plots  
- Optionally follow styles used in prior literature (e.g., Ye 2020 TAC, Science 2016)

---

# Reference
[1] M. Ye, J. Liu, L. Wang, B. D. O. Anderson, and M. Cao,  
“Consensus and Disagreement of Heterogeneous Belief Systems in Influence Networks,”  
*IEEE Transactions on Automatic Control*, vol. 65, no. 11, pp. 4679–4694, 2020.  
Available: https://ieeexplore.ieee.org/document/8941271/
