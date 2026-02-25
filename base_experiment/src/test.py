import torch
import numpy as np
import cirq
import os
import subprocess
from datetime import datetime
import argparse
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from DQL import DQL
from utils.circuit_utils import (
    create_maxcut_circuit,
    calculate_maxcut_objective,
    generate_random_graph,
    expectation_and_best_of_samples,
    greedy_maxcut,
    local_search_maxcut,
)

def run_qaoa(num_qubits, graph_edges, num_trials=100):
    """Run QAOA with fixed gamma=beta=pi/4; return mean Max-Cut expectation (expected edges cut)."""
    simulator = cirq.Simulator()
    total_objective = 0
    params = [np.pi / 4, np.pi / 4]
    for _ in range(num_trials):
        circuit = create_maxcut_circuit(num_qubits, graph_edges, params)
        result = simulator.simulate(circuit)
        total_objective += calculate_maxcut_objective(result, graph_edges)
    return total_objective / num_trials


def run_qaoa_optimized(num_qubits, graph_edges, num_trials=100, optimize_trials=20):
    """Run QAOA with classically optimized gamma, beta; return mean Max-Cut expectation."""
    simulator = cirq.Simulator()

    def neg_mean_expectation(x):
        gamma, beta = x[0], x[1]
        params = [gamma, beta]
        total = 0
        for _ in range(optimize_trials):
            circuit = create_maxcut_circuit(num_qubits, graph_edges, params)
            result = simulator.simulate(circuit)
            total += calculate_maxcut_objective(result, graph_edges)
        return -total / optimize_trials

    res = minimize(
        neg_mean_expectation,
        x0=[np.pi / 4, np.pi / 4],
        method="L-BFGS-B",
        bounds=[(0, 2 * np.pi), (0, 2 * np.pi)],
    )
    best_params = res.x
    total_objective = 0
    for _ in range(num_trials):
        circuit = create_maxcut_circuit(num_qubits, graph_edges, best_params)
        result = simulator.simulate(circuit)
        total_objective += calculate_maxcut_objective(result, graph_edges)
    return total_objective / num_trials


def test_model_on_graph(model, graph_edges, num_qubits, num_samples=100):
    """
    Run trained DQL on the given graph. Resets env to this graph, runs one episode,
    then computes expectation and best-of-K samples from the final circuit state.
    Returns (expectation, best_of_K, total_reward).
    """
    state, _ = model.env.reset(options={"graph_edges": graph_edges})
    total_reward = 0
    done = False
    while not done:
        action = model.select_action(state, model.epsilon_end)
        state, reward, terminated, truncated, _ = model.env.step(action)
        total_reward += reward
        done = terminated or truncated
    exp, best = expectation_and_best_of_samples(
        model.env.circuit, graph_edges, num_samples=num_samples
    )
    return exp, best, total_reward

def create_tmux_session():
    """Create a new tmux session for running the test."""
    session_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create new tmux session
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name])
    
    # Send the command to run test
    cmd = f"cd {os.getcwd()} && CUDA_LAUNCH_BLOCKING=1 python3 src/test.py --tmux"
    subprocess.run(['tmux', 'send-keys', '-t', session_name, cmd, 'C-m'])
    
    print(f"Created tmux session '{session_name}'")
    print("To attach to the session, run: tmux attach -t", session_name)
    print("To detach from the session, press Ctrl+B then D")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmux', action='store_true', help='Indicates if running in tmux session')
    parser.add_argument('--num-graphs', type=int, default=50, help='Number of test graphs')
    parser.add_argument('--no-optimized-qaoa', action='store_true', help='Skip optimized QAOA baseline (faster)')
    parser.add_argument('--no-classical', action='store_true', help='Skip classical baselines (greedy, local search)')
    parser.add_argument('--optimize-params', action='store_true', help='Use DQL that optimizes (gamma,beta) instead of SWAPs; uses models/dql_model_params.pth')
    args = parser.parse_args()

    if not args.tmux:
        # Only create tmux session if not already in one
        create_tmux_session()
        return

    # Hyperparameters from the JSON file
    hyperparams = {
        "num_qubits": 4,
        "gamma": 0.9,
        "max_episodes": 10000,
        "max_steps": 50,
        "edge_probability": 0.2,
        "batch_size": 128,
        "buffer_size": 300000,
        "max_grad_norm": 0.6,
        "epsilon_start": 1.0,
        "epsilon_exploration": 0.8,
        "epsilon_end": 0.15,
        "gae_lambda": 0.9716657825763637,
        "learning_rate": 2.4574373017567988e-05,
        "clip_ratio": 0.25345475557075536,
        "target_kl": 0.045171048250390214,
        "entropy_coef": 0.7343285511655955,
        "entropy_decay": 0.9991448715868556,
        "epsilon_decay": 0.9992954038889231,
        "tau": 0.006076563585289057
    }
    
    # Initialize wandb
    wandb.init(
        project="quantum-rl-dql-test",
        config=hyperparams,
        name=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        reinit=True
    )
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/dql_model_params.pth' if args.optimize_params else 'models/dql_model.pth'
    
    dql = DQL(
        num_qubits=hyperparams["num_qubits"],
        learning_rate=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        max_episodes=hyperparams["max_episodes"],
        max_steps=hyperparams["max_steps"],
        edge_probability=hyperparams["edge_probability"],
        batch_size=hyperparams["batch_size"],
        buffer_size=hyperparams["buffer_size"],
        max_grad_norm=hyperparams["max_grad_norm"],
        epsilon_start=hyperparams["epsilon_start"],
        epsilon_exploration=hyperparams["epsilon_exploration"],
        epsilon_end=hyperparams["epsilon_end"],
        gae_lambda=hyperparams["gae_lambda"],
        clip_ratio=hyperparams["clip_ratio"],
        target_kl=hyperparams["target_kl"],
        entropy_coef=hyperparams["entropy_coef"],
        entropy_decay=hyperparams["entropy_decay"],
        epsilon_decay=hyperparams["epsilon_decay"],
        tau=hyperparams["tau"],
        optimize_params=args.optimize_params,
    )
    
    # Train or load model
    if os.path.exists(model_path):
        print("Loading existing model...")
        dql.q_network.load_state_dict(torch.load(model_path))
        dql.target_q_network.load_state_dict(dql.q_network.state_dict())
    else:
        print("Training new model...")
        dql.train()
        # Save the trained model
        torch.save(dql.q_network.state_dict(), model_path)
    
    num_test_graphs = args.num_graphs
    n_qubits = hyperparams["num_qubits"]
    num_samples = 100  # for "best of K" comparison to classical single-solution
    print(f"\nTesting on {num_test_graphs} random graphs...")
    print("(Quantum: reporting both E[cut] and 'best of {} samples' for fair comparison to classical.)".format(num_samples))

    qaoa_scores = []
    qaoa_best_scores = []   # best cut among num_samples (apples-to-apples with local search)
    dql_scores = []
    dql_best_scores = []
    qaoa_opt_scores = []
    greedy_scores = []
    local_search_scores = []

    for i in tqdm(range(num_test_graphs)):
        test_graph = generate_random_graph(n_qubits, hyperparams["edge_probability"])

        qaoa_score = run_qaoa(n_qubits, test_graph)
        qaoa_scores.append(qaoa_score)
        # One QAOA run, sample K times: "best solution in K shots" (fair vs classical)
        circ = create_maxcut_circuit(n_qubits, test_graph, [np.pi / 4, np.pi / 4])
        _, qaoa_best = expectation_and_best_of_samples(circ, test_graph, num_samples=num_samples)
        qaoa_best_scores.append(qaoa_best)

        dql_exp, dql_best, dql_reward = test_model_on_graph(dql, test_graph, n_qubits, num_samples=num_samples)
        dql_scores.append(dql_exp)
        dql_best_scores.append(dql_best)

        if not args.no_optimized_qaoa:
            qaoa_opt_scores.append(run_qaoa_optimized(n_qubits, test_graph))
        if not args.no_classical:
            greedy_scores.append(greedy_maxcut(n_qubits, test_graph))
            local_search_scores.append(local_search_maxcut(n_qubits, test_graph))

        log_dict = {
            "test_graph": i,
            "qaoa_score": qaoa_score,
            "qaoa_best_of_K": qaoa_best,
            "dql_score": dql_exp,
            "dql_best_of_K": dql_best,
            "improvement_pct": ((dql_exp - qaoa_score) / (qaoa_score + 1e-8)) * 100,
        }
        if not args.no_optimized_qaoa:
            log_dict["qaoa_optimized_score"] = qaoa_opt_scores[-1]
        if not args.no_classical:
            log_dict["greedy_score"] = greedy_scores[-1]
            log_dict["local_search_score"] = local_search_scores[-1]
        wandb.log(log_dict)

    # Statistics (all in expected edges cut)
    qaoa_mean, qaoa_std = np.mean(qaoa_scores), np.std(qaoa_scores)
    dql_mean, dql_std = np.mean(dql_scores), np.std(dql_scores)
    improvement_pct = ((dql_mean - qaoa_mean) / (qaoa_mean + 1e-8)) * 100

    wandb.log({
        "final_qaoa_mean": qaoa_mean,
        "final_qaoa_std": qaoa_std,
        "final_dql_mean": dql_mean,
        "final_dql_std": dql_std,
        "final_improvement_pct": improvement_pct,
    })
    if not args.no_optimized_qaoa:
        wandb.log({
            "final_qaoa_optimized_mean": np.mean(qaoa_opt_scores),
            "final_qaoa_optimized_std": np.std(qaoa_opt_scores),
        })
    if not args.no_classical:
        wandb.log({
            "final_greedy_mean": np.mean(greedy_scores),
            "final_local_search_mean": np.mean(local_search_scores),
        })
    wandb.log({
        "final_qaoa_best_of_K_mean": np.mean(qaoa_best_scores),
        "final_dql_best_of_K_mean": np.mean(dql_best_scores),
    })

    print("\n--- E[cut] (expectation; quantum reports an average over all outcomes) ---")
    print(f"QAOA (fixed γ=β=π/4) - Mean: {qaoa_mean:.4f} ± {qaoa_std:.4f}")
    if not args.no_optimized_qaoa:
        print(f"QAOA (optimized)      - Mean: {np.mean(qaoa_opt_scores):.4f} ± {np.std(qaoa_opt_scores):.4f}")
    print(f"DQL (final-state)     - Mean: {dql_mean:.4f} ± {dql_std:.4f}")
    print(f"Improvement (DQL vs QAOA): {improvement_pct:.2f}%")

    print("\n--- Best of {} samples (fair comparison: one 'solution' per algorithm) ---".format(num_samples))
    print(f"QAOA best-of-{num_samples}  - Mean: {np.mean(qaoa_best_scores):.4f}")
    print(f"DQL best-of-{num_samples}   - Mean: {np.mean(dql_best_scores):.4f}")
    if not args.no_classical:
        print(f"Greedy                 - Mean: {np.mean(greedy_scores):.4f}")
        print(f"Local search            - Mean: {np.mean(local_search_scores):.4f}")
    print("\nNote: Local search returns one solution per run. QAOA/DQL report E[cut] (≤ max cut).")
    print("      'Best of K' = best cut among K measurements — comparable to classical single-solution.")

    # Colors: dark red QAOA exp, red QAOA best, pink QAOA opt, dark blue DQL exp, light blue DQL best, green greedy, black local search
    COLORS = {
        "qaoa_exp": "#8B0000",
        "qaoa_best": "#E63946",
        "qaoa_opt": "#FFB6C1",
        "dql_exp": "#00008B",
        "dql_best": "#87CEEB",
        "greedy": "#228B22",
        "local_search": "#000000",
    }

    # Build list of (label, scores, color) for bar order
    bar_series = [
        ("QAOA E[cut]", qaoa_scores, COLORS["qaoa_exp"]),
        (f"QAOA best-of-{num_samples}", qaoa_best_scores, COLORS["qaoa_best"]),
    ]
    if not args.no_optimized_qaoa:
        bar_series.append(("QAOA (opt) E[cut]", qaoa_opt_scores, COLORS["qaoa_opt"]))
    bar_series.extend([
        ("DQL E[cut]", dql_scores, COLORS["dql_exp"]),
        (f"DQL best-of-{num_samples}", dql_best_scores, COLORS["dql_best"]),
    ])
    if not args.no_classical:
        bar_series.append(("Greedy", greedy_scores, COLORS["greedy"]))
        bar_series.append(("Local search", local_search_scores, COLORS["local_search"]))
    n_series = len(bar_series)
    bar_width = 0.8 / n_series
    x = np.arange(num_test_graphs)

    # Left: per-graph grouped bars
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for i, (label, scores, color) in enumerate(bar_series):
        offset = (i - (n_series - 1) / 2) * bar_width
        plt.bar(x + offset, scores, width=bar_width, label=label, color=color, edgecolor="white", linewidth=0.5)
    plt.xlabel("Test graph index")
    plt.ylabel("Edges cut")
    plt.title("Per-graph: E[cut] vs best-of-K vs classical (single solution)")
    plt.legend()
    plt.ylim(bottom=0)

    # Right: score distribution with side-by-side bars in each bin
    plt.subplot(1, 2, 2)
    all_scores = qaoa_scores + dql_scores + qaoa_best_scores + dql_best_scores
    if not args.no_optimized_qaoa:
        all_scores = all_scores + qaoa_opt_scores
    if not args.no_classical:
        all_scores = all_scores + greedy_scores + local_search_scores
    lo, hi = min(all_scores), max(all_scores)
    margin = (hi - lo) * 0.05 or 0.5
    bin_edges = np.linspace(lo - margin, hi + margin, 25)
    bin_width = (bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1
    bar_width_dist = (bin_width / n_series) * 0.85

    for i, (label, scores, color) in enumerate(bar_series):
        counts, _ = np.histogram(scores, bins=bin_edges)
        densities = counts / (len(scores) * bin_width) if len(scores) else counts
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        offset = (i - (n_series - 1) / 2) * bar_width_dist
        plt.bar(bin_centers + offset, densities, width=bar_width_dist, label=label, color=color, edgecolor="white", linewidth=0.3)
    plt.xlabel("Edges cut")
    plt.ylabel("Density")
    plt.title("Score distribution")
    plt.legend()
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("algorithm_comparison.png")
    wandb.log({"algorithm_comparison": wandb.Image("algorithm_comparison.png")})
    plt.close()
    wandb.finish()

if __name__ == "__main__":
    main() 