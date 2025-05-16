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

To train the model:
```bash
python src/train.py
```

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