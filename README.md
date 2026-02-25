# Quantum Circuit Optimization with RL

This project implements a reinforcement learning approach to optimize quantum circuits for solving the Max-Cut problem. The agent learns to rearrange quantum gates to maximize the solution quality.

## Project Structure

```
.
├── requirements.txt
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   └── quantum_env.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── actor.py
│   │   └── critic.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── circuit_utils.py
│   └── train.py
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

### Running the evaluation experiments (test script)

From the project root, run the test script. It will train a DQL model if none is saved, then evaluate it on random test graphs and compare against QAOA (fixed and optimized) and classical baselines. All scores use the **same metric**: expected number of edges cut (Max-Cut expectation).

**Full run (all baselines, 50 test graphs):**
```bash
python src/test.py --tmux
```
Without `--tmux`, the script starts a detached tmux session and exits; attach with `tmux attach -t test_<timestamp>` to watch. To run in the foreground instead of tmux, run the script once with `--tmux` so it does not spawn tmux again:
```bash
# Run in foreground (no tmux)
python src/test.py --tmux
```

**Faster runs (fewer graphs or skip heavier baselines):**
```bash
# Fewer test graphs
python src/test.py --tmux --num-graphs 20

# Skip optimized QAOA (saves scipy optimization per graph)
python src/test.py --tmux --no-optimized-qaoa

# Skip classical baselines (greedy, local search)
python src/test.py --tmux --no-classical

# Minimal run: 10 graphs, only fixed QAOA vs DQL
python src/test.py --tmux --num-graphs 10 --no-optimized-qaoa --no-classical
```

**DQL that optimizes QAOA (γ, β) instead of SWAPs (theoretically stronger):**
```bash
python src/test.py --tmux --optimize-params
```
Uses `models/dql_model_params.pth`. The agent adjusts gamma and beta each step instead of appending SWAPs, so it directly improves the QAOA ansatz.

**Outputs:**
- Console: mean ± std of expected edges cut for QAOA (fixed), QAOA (optimized), Greedy, Local search, and DQL; improvement % of DQL over fixed QAOA.
- `algorithm_comparison.png`: scatter (per-graph scores) and histogram (distribution), unified y-axis “Expected edges cut”.
- Weights & Biases: project `quantum-rl-dql-test` (if configured).

**Saved model:** If `models/dql_model.pth` exists, training is skipped and the script loads the model and runs evaluation only.

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