import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import torch.distributions as dist

from environment.quantum_env import QuantumCircuitEnv
from models.actor import Actor
from models.critic import Critic

class PPO:
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
        epsilon_exploration=0.3,  # Increased initial exploration
        epsilon_decay=0.999999,  # Slower decay for exploration
        edge_probability=0.20,
        max_steps=50,
        batch_size=128,  # Added batch size for more stable updates
        value_loss_coef=0.5,  # Added value loss coefficient
        max_grad_norm=0.6  # Added gradient clipping
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.env = QuantumCircuitEnv(
            num_qubits=num_qubits,
            edge_probability=edge_probability,
            max_steps=max_steps
        )
        self.num_actions = self.env.action_space.n

        self.actor = Actor(num_qubits, self.num_actions).to(self.device)
        self.critic = Critic(num_qubits).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_iters = train_iters
        self.max_episodes = max_episodes
        self.reward_weight = reward_weight
        self.entropy_coef = entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.entropy_decay = entropy_decay
        self.epsilon_exploration = epsilon_exploration
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        self.episode_rewards = []
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        
        # Initialize wandb
        wandb.init(
            project="quantum-rl",
            config={
                "num_qubits": num_qubits,
                "edge_probability": edge_probability,
                "max_steps": max_steps,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_ratio": clip_ratio,
                "target_kl": target_kl,
                "train_iters": train_iters,
                "max_episodes": max_episodes,
                "reward_weight": reward_weight,
                "entropy_coef": entropy_coef,
                "entropy_coef_min": entropy_coef_min,
                "entropy_decay": entropy_decay,
                "epsilon_exploration": epsilon_exploration,
                "epsilon_decay": epsilon_decay,
                "batch_size": batch_size,
                "value_loss_coef": value_loss_coef,
                "max_grad_norm": max_grad_norm
            }
        )

    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation with reward normalization."""
        advantages = []
        gae = 0
        
        # Normalize rewards
        rewards = np.array(rewards) * self.reward_weight
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values, dtype=torch.float32)
        
        return advantages, returns

    def train_epoch(self, states, actions, old_log_probs, advantages, returns):
        states = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in states.items()}
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.train_iters):
            # Get action probabilities and create distribution
            action_logits = self.actor(states)
            action_probs = torch.softmax(action_logits, dim=-1)
            
            if torch.isnan(action_probs).any():
                print("Warning: NaN values detected in action probabilities")
                action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]
            
            action_dist = dist.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            values = self.critic(states).squeeze()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # Calculate entropy and policy loss
            entropy = action_dist.entropy().mean()
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping
            value_pred_clipped = values + torch.clamp(values - returns, -self.clip_ratio, self.clip_ratio)
            value_loss = torch.max(
                nn.MSELoss()(values, returns),
                nn.MSELoss()(value_pred_clipped, returns)
            ).mean()
            
            # Combined loss
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            # Early stopping based on KL divergence
            kl = (old_log_probs - new_log_probs).mean()
            if kl > 1.5 * self.target_kl:
                break
            
            # Log detailed metrics
            wandb.log({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "kl_divergence": kl.item(),
                "advantage_mean": advantages.mean().item(),
                "advantage_std": advantages.std().item(),
                "current_entropy_coef": self.entropy_coef,
                "current_epsilon": self.epsilon_exploration
            })

    def train(self):
        for episode in tqdm(range(self.max_episodes)):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            unique_actions = set()
            
            states = {k: [] for k in state.keys()}
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []
            
            while not done:
                state_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device) for k, v in state.items()}
                
                # Epsilon-greedy exploration with decay
                if np.random.random() < self.epsilon_exploration:
                    action = np.random.randint(self.num_actions)
                    action_tensor = torch.tensor(action, dtype=torch.long).to(self.device)
                    uniform_probs = torch.ones(self.num_actions, device=self.device) / self.num_actions
                    action_dist = dist.Categorical(uniform_probs)
                    log_prob = action_dist.log_prob(action_tensor)
                else:
                    action_logits = self.actor(state_tensor)
                    action_probs = torch.softmax(action_logits, dim=-1)
                    
                    if torch.isnan(action_probs).any():
                        print("Warning: NaN values detected in action probabilities")
                        action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]
                    
                    action_dist = dist.Categorical(action_probs)
                    action_tensor = action_dist.sample()
                    log_prob = action_dist.log_prob(action_tensor)
                    action = action_tensor.item()
                
                value = self.critic(state_tensor)
                
                try:
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                except Exception as e:
                    print(f"Error in env.step with action {action}: {e}")
                    raise
                
                done = terminated or truncated
                
                for k, v in state.items():
                    states[k].append(v)
                actions.append(action)
                unique_actions.add(action)
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob.item())
                dones.append(done)
                
                state = next_state
                episode_reward += reward
            
            self.episode_rewards.append(episode_reward)
            
            # Update exploration parameters
            self.epsilon_exploration *= self.epsilon_decay
            self.entropy_coef = max(self.entropy_coef_min, self.entropy_coef * self.entropy_decay)
            
            # Track best reward and check for improvement
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            final_value = self.critic(
                {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device) for k, v in state.items()}
            ).item()
            
            advantages, returns = self.compute_gae(
                rewards, values, final_value, dones
            )
            
            self.train_epoch(states, actions, log_probs, advantages, returns)
            
            # Log episode metrics
            wandb.log({
                "episode_reward": episode_reward,
                "episode": episode + 1,
                "average_reward": np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward,
                "best_reward": self.best_reward,
                "unique_actions": len(unique_actions),
                "epsilon_exploration": self.epsilon_exploration,
                "no_improvement_count": self.no_improvement_count
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
    ppo = PPO(
        num_qubits=4,
        edge_probability=0.2,
        max_steps=50,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        target_kl=0.015,
        train_iters=10,
        max_episodes=10000,
        reward_weight=1.0,
        entropy_coef=0.2,
        entropy_coef_min=0.01,
        entropy_decay=0.995,
        epsilon_exploration=0.2,
        epsilon_decay=0.995,
        batch_size=64,
        value_loss_coef=0.5,
        max_grad_norm=0.5
    )
    ppo.train()
    ppo.plot_rewards()
    wandb.finish()