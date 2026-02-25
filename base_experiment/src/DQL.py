import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from environment.quantum_env import QuantumCircuitEnv  # Assuming this is your environment

class Critic(nn.Module):
    def __init__(self, num_qubits, num_actions):
        super(Critic, self).__init__()
        
        state_size = 2**num_qubits * 2  # Real and imaginary parts for quantum state
        
        self.network = nn.Sequential(
            nn.Linear(state_size + 2, 256),  # +2 for circuit_depth and reward
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)  # Output Q-values for each action
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.414)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Args:
            state: dict
                - circuit_state: Tensor of shape (batch_size, 2^num_qubits, 2)
                - circuit_depth: Tensor of shape (batch_size, 1)
                - reward: Tensor of shape (batch_size, 1)
        Output:
            value: Tensor of shape (batch_size, num_actions)
        """
        circuit_state = state['circuit_state'].view(state['circuit_state'].size(0), -1)
        
        x = torch.cat([
            circuit_state,
            state['circuit_depth'],
            state['reward']
        ], dim=1)
        
        value = self.network(x)
        
        return value

class DQL:
    def __init__(
        self,
        num_qubits=4,
        learning_rate=1e-4,  # Increased learning rate
        gamma=0.85,
        gae_lambda=0.999,  # Adjusted for better advantage estimation
        clip_ratio=0.2,
        target_kl=0.02,  # Slightly increased to allow more exploration
        train_iters=20,
        max_episodes=10000,
        reward_weight=1.0,  # Normalized reward weight
        entropy_coef=0.25,  # Increased entropy coefficient for better exploration
        entropy_coef_min=0.01,
        entropy_decay=0.999999,
        epsilon_exploration=0.6,  # Increased initial exploration
        epsilon_decay=0.999999,  # Slower decay for exploration
        edge_probability=0.20,
        max_steps=50,
        batch_size=128,  # Added batch size for more stable updates
        value_loss_coef=0.5,  # Added value loss coefficient
        max_grad_norm=0.6,  # Added gradient clipping
        epsilon_start=1.0,
        epsilon_end=0.01,
        buffer_size=100000,
        tau=0.005,
        optimize_params=False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimize_params = optimize_params
        print(f"Using device: {self.device}")
        if optimize_params:
            print("Env mode: optimizing QAOA (gamma, beta); no SWAPs.")
        
        # Initialize environment
        self.env = QuantumCircuitEnv(
            num_qubits=num_qubits,
            edge_probability=edge_probability,
            max_steps=max_steps,
            optimize_params=optimize_params,
        )
        self.num_actions = self.env.action_space.n
        print(f"Number of actions: {self.num_actions}")
        
        # Initialize Q-networks
        self.q_network = Critic(num_qubits, self.num_actions).to(self.device)
        self.target_q_network = Critic(num_qubits, self.num_actions).to(self.device)
        
        # Copy weights to target network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(param.data)
        
        # Optimizer
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.reward_weight = reward_weight
        self.max_steps = max_steps
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon_exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Tracking
        self.episode_rewards = []
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        
        # Initialize wandb
        wandb.init(
            project="quantum-rl-dql",
            config={
                "num_qubits": num_qubits,
                "edge_probability": edge_probability,
                "max_steps": max_steps,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "tau": tau,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "max_episodes": max_episodes,
                "reward_weight": reward_weight,
                "max_grad_norm": max_grad_norm,
                "epsilon_start": epsilon_start,
                "epsilon_end": epsilon_end,
                "epsilon_decay": epsilon_decay
            }
        )

    def store_transition(self, state, action, reward, next_state, done):
        # Validate action
        if not (0 <= action < self.num_actions):
            print(f"Invalid action: {action}, expected range [0, {self.num_actions-1}]")
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def sample_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert states to dictionary of tensors
        states_dict = {
            'circuit_state': torch.tensor(np.array([s['circuit_state'] for s in states]), dtype=torch.float32).to(self.device),
            'circuit_depth': torch.tensor(np.array([s['circuit_depth'] for s in states]), dtype=torch.float32).to(self.device),
            'reward': torch.tensor(np.array([s['reward'] for s in states]), dtype=torch.float32).to(self.device)
        }
        next_states_dict = {
            'circuit_state': torch.tensor(np.array([s['circuit_state'] for s in next_states]), dtype=torch.float32).to(self.device),
            'circuit_depth': torch.tensor(np.array([s['circuit_depth'] for s in next_states]), dtype=torch.float32).to(self.device),
            'reward': torch.tensor(np.array([s['reward'] for s in next_states]), dtype=torch.float32).to(self.device)
        }
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device) * self.reward_weight
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        return states_dict, actions, rewards, next_states_dict, dones

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        else:
            state_dict = {
                'circuit_state': torch.tensor(state['circuit_state'], dtype=torch.float32).unsqueeze(0).to(self.device),
                'circuit_depth': torch.tensor(state['circuit_depth'], dtype=torch.float32).unsqueeze(0).to(self.device),
                'reward': torch.tensor(state['reward'], dtype=torch.float32).unsqueeze(0).to(self.device)
            }
            with torch.no_grad():
                q_values = self.q_network(state_dict)
                return q_values.argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.sample_batch()
        
        
        q_values = self.q_network(states)  # Shape: [batch_size, num_actions]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states)  # Shape: [batch_size, num_actions]
            next_q_values = next_q_values.max(dim=1)[0]  # Shape: [batch_size]
        
        target_q = rewards + self.gamma * (1 - dones) * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(q_values, target_q)
        
        # Optimize Q-network
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.q_optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Log metrics
        wandb.log({
            "q_loss": loss.item(),
            "average_q_value": q_values.mean().item(),
            "average_next_q_value": next_q_values.mean().item(),
            "epsilon": self.epsilon
        })

    def train(self):
        for episode in tqdm(range(self.max_episodes)):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            unique_actions = set()
            step = 0
            
            # Debug first episode
            if episode == 0:
                print(f"Initial state: {state}")
            
            while not done:
                action = self.select_action(state, self.epsilon)
                
                try:
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                except Exception as e:
                    print(f"Error in env.step with action {action}: {e}")
                    raise
                
                done = terminated or truncated
                self.store_transition(state, action, reward, next_state, done)
                
                # Debug first transition
                if episode == 0 and step == 0:
                    print(f"Transition: action={action}, reward={reward}, next_state={next_state}, done={done}")
                
                state = next_state
                episode_reward += reward
                unique_actions.add(action)
                step += 1
                
                self.train_step()
            
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            self.episode_rewards.append(episode_reward)
            
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            wandb.log({
                "episode_reward": episode_reward,
                "episode": episode + 1,
                "average_reward": np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward,
                "best_reward": self.best_reward,
                "unique_actions": len(unique_actions),
                "no_improvement_count": self.no_improvement_count,
                "episode_steps": step
            })

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('training_rewards.png')
        plt.close()
        
        wandb.log({"training_rewards_plot": wandb.Image('training_rewards.png')})

if __name__ == "__main__":
    dql = DQL(
        num_qubits=4,
        learning_rate=3e-5,  # Increased learning rate
        gamma=.90,
        gae_lambda=0.999,  # Adjusted for better advantage estimation
        clip_ratio=0.6,
        target_kl=0.03,  # Slightly increased to allow more exploration
        train_iters=20,
        max_episodes=30000,
        reward_weight=1.0,  # Normalized reward weight
        entropy_coef=0.55,  # Increased entropy coefficient for better exploration
        entropy_coef_min=0.05,
        entropy_decay=0.9995,
        epsilon_exploration=0.8,  # Increased initial exploration
        epsilon_decay=0.9998,  # Slower decay for exploration
        edge_probability=0.20,
        max_steps=50,
        batch_size=128,  # Added batch size for more stable updates
        value_loss_coef=0.5,  # Added value loss coefficient
        max_grad_norm=0.6,  # Added gradient clipping
        epsilon_start=1.0,
        epsilon_end=0.15,
        buffer_size=300000,
        tau=0.01,
    )
    dql.train()
    dql.plot_rewards()
    wandb.finish()