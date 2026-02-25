# Quantum Circuit Optimization for Max-Cut via Reinforcement Learning: Research Report (Draft)

**Working title:** *Deep Q-Learning for Quantum Circuit Optimization: A Comparison with QAOA on the Max-Cut Problem*

---

## 1. Premise

### 1.1 Context

Quantum approximate optimization algorithms (QAOA) are a leading paradigm for tackling combinatorial optimization on near-term quantum devices. For problems such as **Max-Cut**, QAOA builds a parameterized quantum circuit (mixing and problem Hamiltonians) and optimizes classical parameters to maximize the expected value of the cost function. A limitation is that the **structure** of the circuit (gate order, connectivity) is typically fixed; only the rotation angles are tuned.

**Reinforcement learning (RL)** offers a complementary approach: an agent can learn to **modify the circuit** over time—e.g., by applying SWAP gates or other discrete operations—to improve solution quality. The hypothesis is that an RL agent can discover gate arrangements that outperform or complement fixed-structure QAOA on the same problem instances.

### 1.2 Problem Formulation

- **Problem:** Max-Cut on undirected graphs. Given a graph \(G = (V, E)\), find a partition of \(V\) into two sets that maximizes the number of edges between the two sets.
- **Quantum encoding:** The cost Hamiltonian is the standard Max-Cut Hamiltonian; the quantum state is prepared by a QAOA-style circuit (Hadamards, problem layers, mixing layers). The **objective** reported is the **expectation value** of the cut size (number of edges cut) under the final state.
- **RL formulation:** The environment is a quantum circuit derived from a QAOA template. The **agent** (Deep Q-Network, DQL) chooses **actions** corresponding to applying SWAP gates between pairs of qubits. Each step yields a **reward** that depends on the current Max-Cut expectation, circuit depth, gate count, and improvement over the previous step. The agent’s goal is to maximize cumulative reward over an episode.

### 1.3 Research Question

Can a DQL agent that sequentially modifies a QAOA-derived circuit (via SWAP gates) achieve **better solution quality** on Max-Cut instances than a **fixed-parameter QAOA baseline** (same problem, same number of qubits), when evaluated on held-out random graphs?

---

## 2. Objective

The project has three main objectives:

1. **Meta-training:** Run a hyperparameter search over DQL (learning rate, \(\gamma\), \(\tau\), \(\epsilon\)-decay, entropy coefficients, etc.) to identify a strong configuration for the given environment (4 qubits, 50 steps, random graphs with fixed edge probability).
2. **Training and persistence:** Train a single DQL agent with the best (or chosen) hyperparameters, and **save the trained model** (e.g. `models/dql_model.pth`) so that evaluation can be run without retraining when the model already exists.
3. **Evaluation and comparison:** On a **set of test graphs** (multiple random graphs, same generation process), compute:
   - **QAOA score:** Mean Max-Cut expectation over multiple trials per graph (fixed QAOA parameters, e.g. \(\gamma=\beta=\pi/4\)).
   - **DQL score:** Performance of the trained policy on the same graphs (e.g. total episode reward or, preferably, final Max-Cut value—see below).
   - **Accuracy / comparison metrics:** Aggregate statistics (mean, std) and relative improvement of DQL over QAOA, plus visualizations (scatter plots, histograms) and logging (e.g. Weights & Biases).

Secondary objectives reflected in the codebase:

- **Reproducibility and automation:** Run training/evaluation in a **tmux** session with `CUDA_LAUNCH_BLOCKING=1` for debugging, and **detach** automatically so long jobs can be monitored separately.
- **Experiment tracking:** Log hyperparameters, episode rewards, learning curves, and test results to **Weights & Biases** (wandb) for comparison across runs.

---

## 3. Experiments Conducted

### 3.1 Environment and Baselines

- **Simulator:** Cirq (state-vector simulation).
- **Graphs:** Random graphs on \(n=4\) vertices; each edge present with probability `edge_probability` (e.g. 0.2). Same distribution for training and test.
- **QAOA:** Single layer with parameters \(\gamma = \beta = \pi/4\). QAOA “score” is the **expectation value** of the Max-Cut objective (expected number of edges cut) under the final state, averaged over 100 trials per graph in the test script.
- **DQL:** Double DQN-style agent with a Critic network that takes the circuit state (real/imag parts of the state vector), circuit depth, and current reward. Actions index qubit pairs for SWAP. Experience replay, target network, \(\epsilon\)-greedy exploration.

### 3.2 Meta-Training (Hyperparameter Search)

- **Script:** `src/meta_train.py`.
- **Design:** Multiple independent runs (e.g. 20–30). For each run:
  - Sample hyperparameters from fixed ranges: e.g. `gae_lambda`, `learning_rate`, `clip_ratio`, `target_kl`, `entropy_coef`, `entropy_decay`, `epsilon_decay`, `tau`. Other settings fixed: `num_qubits=4`, `max_episodes=10000`, `max_steps=50`, `edge_probability=0.2`, `batch_size=128`, `buffer_size=300000`, etc.
  - Train DQL for the full number of episodes.
  - Record: best reward, average reward (e.g. over last 100 episodes), final \(\epsilon\), learning progress (e.g. rolling average reward at checkpoints).
- **Outputs:** Per-run JSON files (e.g. `meta_training_results_<timestamp>/run_<id>.json`), CSV of all runs, learning-curve plots, and wandb logs for the project `quantum-rl-dql-hp-search`.
- **Execution:** Can be launched in a new tmux session that runs `python3 src/meta_train.py --tmux` so the process detaches.

### 3.3 Training a Single Model (Best Hyperparameters)

- **Script:** `src/test.py` (or a dedicated train script). Hyperparameters can be loaded from a chosen meta-training run (e.g. `run_13.json`).
- **Behavior:**
  - If `models/dql_model.pth` **exists:** load the saved Q-network (and target) and skip training.
  - If **not:** train DQL with the given hyperparameters, then save the Q-network state dict to `models/dql_model.pth`.
- This allows “train once, evaluate many times” and avoids recomputing when only evaluation or comparison is needed.

### 3.4 Evaluation: QAOA vs DQL on Test Graphs

- **Script:** `src/test.py` (when run with `--tmux` or directly).
- **Test set:** \(N\) random graphs (e.g. \(N=50\)), same \(n=4\) and `edge_probability`.
- **Per graph:**
  - **QAOA:** Build QAOA circuit with \(\gamma=\beta=\pi/4\), simulate 100 times, compute Max-Cut expectation for each, report **mean** → one “QAOA score” per graph (in **Max-Cut expectation units**).
  - **DQL:** Reset the environment with that graph, run the trained policy (e.g. \(\epsilon = \epsilon_{\text{end}}\)) for one episode, sum rewards over steps → one “DQL score” per graph (in **cumulative reward units**).
- **Aggregate:** Mean and standard deviation of QAOA scores and DQL scores across the \(N\) graphs; “improvement” as \(({\text{DQL}} - {\text{QAOA}}) / {\text{QAOA}} \times 100\%\) (note: this mixes reward vs Max-Cut scale; see Section 5).
- **Artifacts:** Scatter plot (scores vs graph index), histograms of score distributions, `algorithm_comparison.png`, and wandb logs (e.g. project `quantum-rl-dql-test`).

### 3.5 Execution and Tooling

- **Tmux:** `test.py` can create a detached tmux session that runs `CUDA_LAUNCH_BLOCKING=1 python3 src/test.py --tmux` so that training/evaluation runs in the background and can be reattached for inspection.
- **Imports:** All entry points assume execution from the project root with `src` on the path (e.g. `python3 src/test.py`); imports use `DQL`, `environment.quantum_env`, `utils.circuit_utils`, etc., consistent with the codebase under `src/`.

---

## 4. Results of the Experiments

### 4.1 Meta-Training

- **Example run (e.g. run_13):** Reported metrics include:
  - **Best episode reward:** ~181.24  
  - **Average reward (e.g. last 100 episodes):** ~176.84  
  - **Final \(\epsilon\):** 0.15  
  - **Total episodes:** 10,000  
  - **Learning progress:** Rolling average reward increases from ~165 to ~177 over the run, indicating that the agent learns to achieve higher cumulative reward over training.
- **Output location:** Results saved under `meta_training_results_<timestamp>/` (e.g. `run_13.json`, `all_results.csv`). Exact numbers and number of runs depend on the executed meta-training campaign.

### 4.2 Test-Set Comparison (QAOA vs DQL)

- **Reported metrics (example):**
  - **QAOA – Mean ± Std:** Mean and standard deviation of the **per-graph mean Max-Cut expectation** (e.g. over 50 graphs).
  - **DQL – Mean ± Std:** Mean and standard deviation of the **per-graph total episode reward** (cumulative reward over the episode).
  - **Improvement:** Percentage difference between DQL mean and QAOA mean, computed as above (with the caveat that units differ).
- **Visualizations:**  
  - Scatter: QAOA and DQL scores vs test graph index.  
  - Histograms: Distribution of QAOA scores and DQL scores.  
  - These are saved as `algorithm_comparison.png` and logged to wandb.

*Note:* Concrete numbers (e.g. “QAOA mean = X, DQL mean = Y”) should be filled in from the actual wandb run or console output of a specific test run; the report structure above is the template for where those values go.

---

## 5. Interpretation of the Results

### 5.1 What the “Scores” Represent (Y-Axis in Plots)

- **QAOA score (y-axis for QAOA):**  
  **Expected number of edges cut** (Max-Cut expectation) for the fixed-parameter QAOA circuit on that graph, averaged over 100 trials. So the y-axis is in **“number of edges”** (or expectation thereof). Higher is better.

- **DQL score (y-axis for DQL):**  
  **Cumulative episode reward** for the trained policy on that graph. The reward at each step is:
  \[
  \text{reward} = 2 \cdot \text{MaxCut} - 0.01 \cdot \text{depth} - 0.005 \cdot \text{gates} + \text{improvement\_bonus} + 0.1.
  \]
  So the DQL “score” is a **sum over up to 50 steps** of this quantity—not the raw Max-Cut value at the end of the episode. The scale is therefore different from QAOA, and the two y-axes are **not directly comparable** as “same quantity.” The plot shows that DQL tends to achieve higher **reward** than the baseline when the baseline is expressed in Max-Cut units; a fairer comparison would use the **same** quantity for both (see Next Steps).

### 5.2 Learning and Robustness

- Meta-training shows that DQL can learn: rolling average reward increases with episodes, and best runs reach notably higher cumulative reward.
- Saving and reusing the trained model allows consistent evaluation across many test graphs and reproducible comparison with QAOA.

### 5.3 Limitations

- **Scale:** Only 4 qubits and 50 steps; graphs are small and episodes short.
- **Baseline:** QAOA uses fixed \(\gamma=\beta=\pi/4\); no classical parameter optimization for QAOA, so the baseline is not “optimized QAOA.”
- **Metric mismatch:** DQL is compared using cumulative reward; QAOA using Max-Cut expectation. So “improvement” is indicative but not an apples-to-apples accuracy or approximation-ratio comparison.

---

## 6. Next Steps: Completing the Research Project

To turn this into a complete, publishable study, the following experiments and refinements are recommended.

### 6.1 Fair Comparison Metric

- **Report Max-Cut for both algorithms:**  
  For each test graph, after the DQL episode ends, compute the **Max-Cut expectation** of the **final circuit state** (same function as for QAOA). Use this as the “DQL score” in addition to (or instead of) cumulative reward for the comparison plots.
- **Unify y-axis:**  
  All comparison plots (scatter, histograms) should use **expected number of edges cut** for both QAOA and DQL so that improvement and variance are comparable.

### 6.2 Broader and More Systematic Experiments

- **Graph set:**  
  - Increase number of test graphs (e.g. 100–500).  
  - Vary graph types: different edge densities, fixed edge count, or structured graphs (e.g. regular, small-world) to test generalization.
- **Scaling:**  
  - Run experiments for \(n = 6, 8\) qubits (and larger if feasible) to see how DQL and QAOA scale.  
  - Consider varying `max_steps` (e.g. 25, 50, 100) to study the effect of episode length.
- **Seeds and statistics:**  
  - Multiple random seeds for training and for graph generation; report mean and confidence intervals (e.g. 95%) for mean score and improvement.

### 6.3 Stronger Baselines

- **Optimized QAOA:**  
  Run QAOA with **classically optimized** \(\gamma, \beta\) (e.g. gradient descent or Bayesian optimization) and compare DQL against this stronger baseline.
- **Other algorithms:**  
  Compare to classical heuristics (e.g. greedy, local search) and, if relevant, to other quantum or hybrid methods (e.g. different QAOA depths, VQE).

### 6.4 Ablations and Analysis

- **Reward design:**  
  Ablate reward terms (depth penalty, gate penalty, improvement bonus) to see which drive learning and whether pure Max-Cut reward is sufficient or harmful.
- **Action space:**  
  Test alternative actions (e.g. gate insertions, parameter updates) in addition to or instead of SWAPs.
- **DQL vs other RL:**  
  Compare with PPO, SAC, or other algorithms already present in the codebase under the same environment and evaluation protocol.

### 6.5 Reproducibility and Dissemination

- **Code and data:**  
  Document how to reproduce meta-training, training, and evaluation (scripts, seeds, config files). Optionally release graphs and saved models.
- **Wandb:**  
  Use one project (or tags) for final experiments so that all runs (meta-training, training, evaluation) are linked and hyperparameters/results are queryable.
- **Paper structure:**  
  Expand this report into full sections: Abstract, Introduction, Related Work, Method (environment, DQL, QAOA baseline), Experiments (protocol, hyperparameters, evaluation metric), Results (tables and figures with fair Max-Cut comparison), Discussion, and Conclusion.

---

## 7. Summary

This project investigates **RL-based quantum circuit optimization** for Max-Cut: a DQL agent modifies a QAOA-derived circuit via SWAP gates to maximize a reward that is heavily influenced by the Max-Cut objective. **Meta-training** is used to select hyperparameters; the **trained model** is saved and reused for evaluation. **Test-set comparison** is performed against a fixed-parameter QAOA baseline on multiple random graphs, with results and learning curves logged to wandb and visualized in scatter plots and histograms.

**Important caveat:** Current “scores” use **Max-Cut expectation** for QAOA and **cumulative reward** for DQL; the y-axis in comparison plots therefore represents different quantities. A critical next step is to report **Max-Cut expectation for both** and to re-run comparison and plots using this unified metric. Further work should scale problem size, diversify graph types, add optimized QAOA and other baselines, and run ablations and multiple seeds to support robust conclusions for a full research paper.
