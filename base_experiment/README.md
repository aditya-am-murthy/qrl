# Base Experiment (4 qubits, SWAP / (γ,β) DQL)

Original experiment: DQL/PPO/SAC with 4 qubits, comparison vs QAOA and classical baselines.

Run from **repository root**:

```bash
# Install dependencies (from repo root)
pip install -r requirements.txt

# Test/evaluation (trains if no model, then evaluates)
python base_experiment/src/test.py --tmux

# Param-optimization mode (DQL tunes γ, β instead of SWAPs)
python base_experiment/src/test.py --tmux --optimize-params
```

See `RESEARCH_REPORT.md` in this folder for the full research report.
