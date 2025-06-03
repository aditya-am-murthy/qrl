import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import torch.distributions as dist

from environment.quantum_env import QuantumCircuitEnv
from models.actor import Actor

class Critic(nn.Module):
    def __init__(self, num_qubits, num_actions):
        super(Critic, self).__init__()
        
        state_size = 2**num_qubits * 2  # Real and imaginary parts
        
        self.network = nn.Sequential(
            nn.Linear(state_size + 2, 256),  # +2 for circuit depth and reward
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)  # Output Q-values for all actions
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
            q_values: Tensor of shape (batch_size, num_actions)
        """
        circuit_state = state['circuit_state'].view(state['circuit_state'].size(0), -1)
        
        x = torch.cat([
            circuit_state,
            state['circuit_depth'],
            state['reward']
        ], dim=1)
        
        q_values = self.network(x)
        
        return q_values

class SAC:
    def __init__(
        self,
        num_qubits=4,
        learning_rate_actor=5e-4,
        learning_rate_critic=3e-4,
        alpha=0.2,  # Entropy coefficient
        gamma=0.99,
        tau=0.005,  # Soft target update rate
        buffer_size=100000,
        batch_size=64,
        max_episodes=10000,
        reward_weight=1.0,
        max_steps=50,
        edge_probability=0.2,
        max_grad_norm=1.0
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize environment
        self.env = QuantumCircuitEnv(
            num_qubits=num_qubits,
            edge_probability=edge_probability,
            max_steps=max_steps
        )
        self.num_actions = self.env.action_space.n

        # Initialize networks
        self.actor = Actor(num_qubits, self.num_actions).to(self.device)
        self.q1 = Critic(num_qubits, self.num_actions).to(self.device)
        self.q2 = Critic(num_qubits, self.num_actions).to(self.device)
        self.q1_target = Critic(num_qubits, self.num_actions).to(self.device)
        self.q2_target = Critic(num_qubits, self.num_actions).to(self.device)
        
        # Copy weights to target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(param.data)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate_critic)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate_critic)
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.reward_weight = reward_weight
        self.max_steps = max_steps
        self.max_grad_norm = max_grad_norm
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Tracking
        self.episode_rewards = []
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        
        # Initialize wandb
        wandb.init(
            project="quantum-rl-sac",
            config={
                "num_qubits": num_qubits,
                "edge_probability": edge_probability,
                "max_steps": max_steps,
                "learning_rate_actor": learning_rate_actor,
                "learning_rate_critic": learning_rate_critic,
                "alpha": alpha,
                "gamma": gamma,
                "tau": tau,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "max_episodes": max_episodes,
                "reward_weight": reward_weight,
                "max_grad_norm": max_grad_norm
            }
        )

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def sample_batch(self):
        """Sample a batch from the replay buffer."""
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert states to tensors (handle dictionary states)
        states = {k: torch.tensor(np.array([s[k] for s in states]), dtype=torch.float32).to(self.device)
                  for k in states[0].keys()}
        next_states = {k: torch.tensor(np.array([s[k] for s in next_states]), dtype=torch.float32).to(self.device)
                       for k in next_states[0].keys()}
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device) * self.reward_weight
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        return states, actions, rewards, next_states, dones

    def train_step(self):
        """Perform one training step on a batch of data."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.sample_batch()
        
        # Actor loss
        action_logits = self.actor(states)
        action_probs = torch.softmax(action_logits, dim=-1)
        action_dist = dist.Categorical(action_probs)
        action_samples = action_dist.sample()
        log_probs = action_dist.log_prob(action_samples)
        
        q1_values = self.q1(states)
        q2_values = self.q2(states)
        q_values = torch.min(q1_values, q2_values)
        actor_loss = (self.alpha * log_probs - q_values.gather(1, action_samples.unsqueeze(1))).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Critic loss
        next_action_logits = self.actor(next_states)
        next_action_probs = torch.softmax(next_action_logits, dim=-1)
        next_action_dist = dist.Categorical(next_action_probs)
        next_action_samples = next_action_dist.sample()  # Shape: [batch_size]
        next_log_probs = next_action_dist.log_prob(next_action_samples)  # Shape: [batch_size]
        
        q1_next = self.q1_target(next_states)  # Shape: [batch_size, num_actions]
        q2_next = self.q2_target(next_states)  # Shape: [batch_size, num_actions]
        q_next = torch.min(q1_next, q2_next)  # Shape: [batch_size, num_actions]
        
        # Select Q-value for sampled next action
        next_q_values = q_next.gather(1, next_action_samples.unsqueeze(1))  # Shape: [batch_size, 1]
        
        # Compute target Q-value
        target_q = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * (next_q_values - self.alpha * next_log_probs.unsqueeze(1))  # Shape: [batch_size, 1]
        
        # Debug shapes
        if not hasattr(self, 'logged_shapes'):
            wandb.log({
                "debug/q1_values_shape": str(q1_values.shape),
                "debug/actions_shape": str(actions.shape),
                "debug/target_q_shape": str(target_q.shape),
                "debug/next_q_values_shape": str(next_q_values.shape),
                "debug/next_log_probs_shape": str(next_log_probs.shape)
            })
            self.logged_shapes = True
        
        # Critic loss
        q1_pred = self.q1(states).gather(1, actions.unsqueeze(1))  # Shape: [batch_size, 1]
        q2_pred = self.q2(states).gather(1, actions.unsqueeze(1))  # Shape: [batch_size, 1]
        
        q1_loss = nn.MSELoss()(q1_pred, target_q.detach())
        q2_loss = nn.MSELoss()(q2_pred, target_q.detach())
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
        self.q2_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Log metrics
        wandb.log({
            "actor_loss": actor_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_entropy": action_dist.entropy().mean().item(),
            "average_q_value": q_values.mean().item()
        })

    def train(self):
        """Train the SAC agent."""
        for episode in tqdm(range(self.max_episodes)):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            unique_actions = set()
            
            step = 0
            while not done:
                state_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
                               for k, v in state.items()}
                
                # Sample action from policy
                with torch.no_grad():
                    action_logits = self.actor(state_tensor)
                    action_probs = torch.softmax(action_logits, dim=-1)
                    if torch.isnan(action_probs).any():
                        print("Warning: NaN values detected in action probabilities")
                        action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]
                    action_dist = dist.Categorical(action_probs)
                    action = action_dist.sample().item()
                
                try:
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                except Exception as e:
                    print(f"Error in env.step with action {action}: {e}")
                    raise
                
                done = terminated or truncated
                self.store_transition(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                unique_actions.add(action)
                step += 1
                
                # Perform training step
                self.train_step()
            
            self.episode_rewards.append(episode_reward)
            
            # Track best reward and early stopping
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            if self.no_improvement_count > 500:
                print(f"Early stopping at episode {episode + 1} due to no improvement")
                break
            
            # Log episode metrics
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
        """Plot and log training rewards."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('training_rewards.png')
        plt.close()
        
        wandb.log({"training_rewards_plot": wandb.Image('training_rewards.png')})

if __name__ == "__main__":
    sac = SAC(
        num_qubits=4,
        learning_rate_actor=3e-5,
        learning_rate_critic=1e-5,
        alpha=0.2,
        gamma=0.95,
        tau=0.01,
        buffer_size=1000000,
        batch_size=64,
        max_episodes=100000,
        reward_weight=1.0,
        max_steps=50,
        edge_probability=0.2,
        max_grad_norm=1.0
    )
    sac.train()
    sac.plot_rewards()
    wandb.finish()