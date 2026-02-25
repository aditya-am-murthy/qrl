# Quantum Circuit Optimization with RL

This project implements a reinforcement learning approach to optimize quantum circuits for solving the Max-Cut problem. The agent learns to rearrange quantum gates to maximize the solution quality.

## Project Structure

```
.
├── requirements.txt
├── base_experiment/          # Original 4-qubit experiment (DQL/PPO/SAC vs QAOA)
│   ├── src/
│   │   ├── environment/ quantum_env.py
│   │   ├── models/ actor.py, critic.py
│   │   ├── utils/ circuit_utils.py
│   │   ├── test.py, meta_train.py, DQL.py, PPO.py, SAC.py
│   ├── RESEARCH_REPORT.md
│   └── README.md
├── time_comparison/          # Time + iteration comparison (8 qubits, cuQuantum, parallel classical)
│   ├── utils/ circuit_utils.py (qsimcirq/cuQuantum)
│   ├── classical/ parallel_runner.py (greedy, local search)
│   ├── qaoa/ qaoa_baseline.py
│   ├── environment/ quantum_env_params.py
│   ├── optimizers/ dql_runner, ppo_runner, sac_runner
│   ├── run_experiment.py
│   └── README.md
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Base experiment (4 qubits)

See `base_experiment/README.md`. From repo root:
```bash
python base_experiment/src/test.py --tmux
```

### Time comparison experiment (8 qubits, timing + iterations)

See `time_comparison/README.md`. Compares wall time and iteration counts for Greedy, Local search, QAOA (classical opt), DQL, PPO, SAC. Uses cuQuantum (qsimcirq) when available.
```bash
pip install -r time_comparison/requirements.txt
python time_comparison/run_experiment.py --num-graphs 20
```

### Base evaluation (test script)

From the project root. It will train a DQL model if none is saved, then evaluate on random test graphs vs QAOA and classical baselines (same metric: expected edges cut).

**Full run (all baselines, 50 test graphs):**
```bash
python base_experiment/src/test.py --tmux
```
Without `--tmux`, the script starts a detached tmux session. To run in the foreground:
```bash
python base_experiment/src/test.py --tmux
```

**Faster runs:**
```bash
python base_experiment/src/test.py --tmux --num-graphs 20
python base_experiment/src/test.py --tmux --no-optimized-qaoa --no-classical
```

**DQL that optimizes QAOA (γ, β) instead of SWAPs (theoretically stronger):**
```bash
python base_experiment/src/test.py --tmux --optimize-params
```
Uses `models/dql_model_params.pth` (saved in cwd when run from repo root). The agent adjusts gamma and beta each step instead of appending SWAPs, so it directly improves the QAOA ansatz.

**Outputs:**
- Console: mean ± std of expected edges cut for QAOA (fixed), QAOA (optimized), Greedy, Local search, and DQL; improvement % of DQL over fixed QAOA.
- `algorithm_comparison.png`: scatter (per-graph scores) and histogram (distribution), unified y-axis “Expected edges cut”.
- Weights & Biases: project `quantum-rl-dql-test` (if configured).

**Saved model:** If `models/dql_model.pth` exists (in cwd), training is skipped.

## Project Components

1. **Quantum Environment**: Implements a custom Gymnasium environment that:
   - Represents quantum circuits as a sequence of gates
   - Provides actions for rearranging gates
   - Calculates rewards based on solution quality

2. **RL Agent**: Uses PPO (Proximal Policy Optimization) with:
   - Actor network for policy
   - Critic network for value estimation
   - Experience replay buffer

3. **Circuit Optimization**: Focuses on:
   - Gate arrangement optimization
   - Circuit depth minimization
   - Solution quality maximization

## Implementation Details

The project uses:
- PyTorch for deep learning
- Qiskit for quantum circuit simulation
- Gymnasium for RL environment
- PPO algorithm for training 