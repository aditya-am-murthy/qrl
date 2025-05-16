import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from environment.quantum_env import QuantumCircuitEnv
from models.actor import Actor
from models.critic import Critic

class PPO:
    def __init__(
        self,
        num_qubits=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        target_kl=0.01,
        train_iters=10,
        max_episodes=1000
    ):
        self.env = QuantumCircuitEnv(num_qubits=num_qubits)
        self.num_actions = self.env.action_space.n

        self.actor = Actor(num_qubits, self.num_actions)
        self.critic = Critic(num_qubits)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_iters = train_iters
        self.max_episodes = max_episodes
        
        self.episode_rewards = []
    
    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
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
    
    def train_epoch(self, states, actions, old_log_probs, advantages, returns):    #one epoch train
        states = {k: torch.tensor(v, dtype=torch.float32) for k, v in states.items()}
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.train_iters):
            new_log_probs = self.actor.get_action(states)[1]
            values = self.critic(states).squeeze()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.MSELoss()(values, returns)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
            
            kl = (old_log_probs - new_log_probs).mean()
            if kl > 1.5 * self.target_kl:
                break
    
    def train(self):
        for episode in tqdm(range(self.max_episodes)):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            states = {k: [] for k in state.keys()}
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []
            
            while not done:
                action, log_prob = self.actor.get_action(
                    {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in state.items()}
                )
                value = self.critic(
                    {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in state.items()}
                )
                
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                
                for k, v in state.items():
                    states[k].append(v)
                actions.append(action.item())
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob.item())
                dones.append(done)
                
                state = next_state
                episode_reward += reward
            
            self.episode_rewards.append(episode_reward)
            
            final_value = self.critic(
                {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in state.items()}
            ).item()
            
            advantages, returns = self.compute_gae(
                rewards, values, final_value, dones
            )
            
            self.train_epoch(states, actions, log_probs, advantages, returns)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    def plot_rewards(self):
        #training reqrd plsots
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('training_rewards.png')
        plt.close()

if __name__ == "__main__":
    ppo = PPO(num_qubits=4)
    ppo.train()
    ppo.plot_rewards() 