# Belief System Dynamics Simulation Framework
A modular platform for generating and simulating belief system dynamics on social influence networks, based on:

**Ye et al., IEEE TAC 2020 — "Consensus and Disagreement of Heterogeneous Belief Systems in Influence Networks".**

The framework produces:
- Social influence matrices $ W $  
- Heterogeneous logic matrices $ C_i $  
- Initial beliefs $ X_0 $

and supports exporting, visualization, and downstream simulation.

---

# Model Overview

Each agent $ i $ holds an $ m $-dimensional belief vector $ x_i(t) $.  
Belief updates follow the extended DeGroot model:

$$
x_i(t+1) = \sum_{j=1}^n W_{ij} \, C_i \, x_j(t).
$$

- $ W $: row-stochastic influence matrix  
- $ C_i $: logic matrix encoding topic dependencies  
- Heterogeneity in $ C_i $ may lead to consensus or persistent disagreement

---

# Repository Structure

### **1. `social_network.py`**
Generates the influence matrix $ W $ using ER, WS, BA, or Random-Regular models.  
- Beta-distributed edge weights  
- Ensures row-stochasticity  
- Provides network summary + Gephi export

### **2. `logic_matrix.py`**
Creates baseline and heterogeneous logic matrices $ C_i $.  
- Lower-triangular structure  
- Beta-distributed coefficients  
- Supports sparsity & heterogeneity  
- Exports baseline $ C_{\text{base}} $

### **3. `init_belief.py`**
Generates initial beliefs $ X_0 $.  
Modes: `uniform`, `beta`.
Exports all results into a timestamped folder.

### **4. `belief_simulation.ipynb`**
Interactive notebook for testing:
- Component generation
- Simple belief dynamics
- Visualizations


# Simulation Pipeline

1. Generate $ W $, $ C_i $, and $ X_0 $  
2. Update beliefs over time using  
   $$
   X(t+1) = \sum_{j=1}^n W_{ij} \, C_i \, x_j(t).
   $$
3. Stop when converged or max steps reached  
4. Visualize belief trajectories and network structure

---

# Reference
[1] M. Ye, J. Liu, L. Wang, B. D. O. Anderson, and M. Cao,
“Consensus and Disagreement of Heterogeneous Belief Systems
in Influence Networks,” IEEE Transactions on Automatic Control,
vol. 65, no. 11, pp. 4679–4694, Nov. 2020.