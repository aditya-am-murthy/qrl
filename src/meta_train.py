import itertools
import json
import os
import random
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import wandb
from DQL import DQL
import matplotlib.pyplot as plt

class HyperparameterSearch:
    def __init__(self, num_runs: int = 20):
        self.num_runs = num_runs
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"meta_training_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

    def generate_hyperparameters(self) -> Dict[str, Any]:
        """Generate a random set of hyperparameters within specified ranges.
        Only randomizes the most impactful hyperparameters while keeping others at default values."""
        return {
            # Fixed hyperparameters
            'num_qubits': 4,
            'gamma': 0.90,
            'max_episodes': 10000,
            'max_steps': 50,
            'edge_probability': 0.20,
            'batch_size': 128,
            'buffer_size': 300000,
            'max_grad_norm': 0.6,
            'epsilon_start': 1.0,
            'epsilon_exploration': 0.8,
            'epsilon_end': 0.15,
            
            # Randomized impactful hyperparameters
            'gae_lambda': random.uniform(0.95, 0.999),
            'learning_rate': random.uniform(1e-5, 1e-3),
            'clip_ratio': random.uniform(0.1, 0.8),
            'target_kl': random.uniform(0.01, 0.05),
            'entropy_coef': random.uniform(0.1, 0.8),
            'entropy_decay': random.uniform(0.999, 0.9999),
            'epsilon_decay': random.uniform(0.999, 0.9999),
            'tau': random.uniform(0.001, 0.02),
        }

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def run_training(self, hyperparams: Dict[str, Any], run_id: int, window_size: int = 100):
        """Run a single training session with given hyperparameters."""
        # Initialize wandb run
        wandb.init(
            project="quantum-rl-dql-hp-search",
            config=hyperparams,
            name=f"run_{run_id}_{self.timestamp}",
            reinit=True
        )

        # Create DQL instance with hyperparameters
        dql = DQL(**hyperparams)
        
        # Track learning progress
        progress_metrics = {
            'episode_rewards': [],
            'rolling_averages': [],  # Will store average rewards over time
            'checkpoint_episodes': []  # Will store episode numbers for each checkpoint
        }
        
        # Train the model
        dql.train()
        
        # Calculate rolling averages at regular intervals
        checkpoint_interval = max(1, dql.max_episodes // 10)  # 10 checkpoints throughout training
        
        for i in range(0, len(dql.episode_rewards), checkpoint_interval):
            end_idx = min(i + window_size, len(dql.episode_rewards))
            if end_idx > i:  # Ensure we have enough episodes for the window
                rolling_avg = np.mean(dql.episode_rewards[i:end_idx])
                progress_metrics['rolling_averages'].append(float(rolling_avg))
                progress_metrics['checkpoint_episodes'].append(end_idx)
        
        # Get final metrics
        final_metrics = {
            'best_reward': dql.best_reward,
            'average_reward': np.mean(dql.episode_rewards[-100:]),
            'final_epsilon': dql.epsilon,
            'total_episodes': len(dql.episode_rewards),
            'learning_progress': progress_metrics
        }
        
        # Log final metrics
        wandb.log(final_metrics)
        
        # Create a learning progress plot
        plt.figure(figsize=(10, 5))
        plt.plot(progress_metrics['checkpoint_episodes'], progress_metrics['rolling_averages'], 
                label='Rolling Average Reward')
        plt.title('Learning Progress')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(self.results_dir, f'learning_progress_run_{run_id}.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Log the plot to wandb
        wandb.log({"learning_progress_plot": wandb.Image(plot_path)})
        
        # Store results
        result = {
            'run_id': run_id,
            'hyperparameters': hyperparams,
            'metrics': final_metrics,
            'timestamp': self.timestamp
        }
        self.results.append(result)
        
        # Convert to serializable format before saving
        serializable_result = self._convert_to_serializable(result)
        
        # Save individual run results
        with open(os.path.join(self.results_dir, f'run_{run_id}.json'), 'w') as f:
            json.dump(serializable_result, f, indent=4)
        
        wandb.finish()
        return result

    def run_all(self):
        """Run all training sessions with different hyperparameters."""
        for i in range(self.num_runs):
            hyperparams = self.generate_hyperparameters()
            print(f"\nStarting run {i+1}/{self.num_runs}")
            print("Hyperparameters:", json.dumps(hyperparams, indent=2))
            
            result = self.run_training(hyperparams, i, hyperparams['max_episodes']//10)
            print(f"Run {i+1} completed. Best reward: {result['metrics']['best_reward']}")
        
        # Save all results to a CSV file
        self.save_results()

    def save_results(self):
        """Save all results to a CSV file for later analysis."""
        # Convert results to DataFrame
        rows = []
        for result in self.results:
            row = {
                'run_id': result['run_id'],
                'timestamp': result['timestamp'],
                **result['metrics'],
                **{f'hp_{k}': v for k, v in result['hyperparameters'].items()}
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.results_dir, 'all_results.csv'), index=False)
        print(f"\nResults saved to {self.results_dir}/all_results.csv")

def create_tmux_session():
    """Create a new tmux session for running the meta-training."""
    session_name = f"meta_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create new tmux session
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name])
    
    # Send the command to run meta-training
    cmd = f"cd {os.getcwd()} && python3 src/meta_train.py --tmux"
    subprocess.run(['tmux', 'send-keys', '-t', session_name, cmd, 'C-m'])
    
    print(f"Created tmux session '{session_name}'")
    print("To attach to the session, run: tmux attach -t", session_name)
    print("To detach from the session, press Ctrl+B then D")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmux', action='store_true', help='Indicates if running in tmux session')
    args = parser.parse_args()

    if not args.tmux:
        # Only create tmux session if not already in one
        create_tmux_session()
    else:
        # Run the search if in tmux session
        search = HyperparameterSearch(num_runs=30)
        search.run_all() 