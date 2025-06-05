import torch
import numpy as np
import cirq
import os
from src.DQL import DQL
from src.utils.circuit_utils import create_maxcut_circuit, calculate_maxcut_objective, generate_random_graph
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_qaoa(num_qubits, graph_edges, num_trials=100):
    """Run QAOA algorithm and return average objective value."""
    simulator = cirq.Simulator()
    total_objective = 0
    
    for _ in range(num_trials):
        # Create QAOA circuit with standard parameters
        params = [np.pi/4, np.pi/4]  # Standard QAOA parameters
        circuit = create_maxcut_circuit(num_qubits, graph_edges, params)
        
        # Simulate circuit
        result = simulator.simulate(circuit)
        
        # Calculate objective value
        objective = calculate_maxcut_objective(result, graph_edges)
        total_objective += objective
    
    return total_objective / num_trials

def test_model_on_graph(model, graph_edges, num_qubits):
    """Test a trained model on a specific graph."""
    state, _ = model.env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = model.select_action(state, model.epsilon_end)
        state, reward, terminated, truncated, _ = model.env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    return total_reward

def main():
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
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    model_path = 'models/dql_model.pth'
    
    # Initialize DQL model
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
        tau=hyperparams["tau"]
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
    
    # Test on multiple graphs
    num_test_graphs = 50
    print(f"\nTesting on {num_test_graphs} random graphs...")
    
    dql_scores = []
    qaoa_scores = []
    
    for i in tqdm(range(num_test_graphs)):
        # Generate test graph
        test_graph = generate_random_graph(hyperparams["num_qubits"], hyperparams["edge_probability"])
        
        # Run QAOA
        qaoa_score = run_qaoa(hyperparams["num_qubits"], test_graph)
        qaoa_scores.append(qaoa_score)
        
        # Run DQL
        dql_score = test_model_on_graph(dql, test_graph, hyperparams["num_qubits"])
        dql_scores.append(dql_score)
    
    # Calculate statistics
    dql_mean = np.mean(dql_scores)
    dql_std = np.std(dql_scores)
    qaoa_mean = np.mean(qaoa_scores)
    qaoa_std = np.std(qaoa_scores)
    
    print("\nResults:")
    print(f"QAOA - Mean: {qaoa_mean:.4f} ± {qaoa_std:.4f}")
    print(f"DQL  - Mean: {dql_mean:.4f} ± {dql_std:.4f}")
    print(f"Improvement: {((dql_mean - qaoa_mean) / qaoa_mean * 100):.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Plot individual scores
    plt.subplot(1, 2, 1)
    plt.scatter(range(num_test_graphs), qaoa_scores, alpha=0.5, label='QAOA')
    plt.scatter(range(num_test_graphs), dql_scores, alpha=0.5, label='DQL')
    plt.xlabel('Test Graph Index')
    plt.ylabel('Score')
    plt.title('Individual Test Scores')
    plt.legend()
    
    # Plot distribution
    plt.subplot(1, 2, 2)
    plt.hist(qaoa_scores, alpha=0.5, label='QAOA', bins=20)
    plt.hist(dql_scores, alpha=0.5, label='DQL', bins=20)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 