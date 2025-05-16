import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network for PPO algorithm.
    Takes quantum circuit state as input and outputs action probabilities.
    """
    
    def __init__(self, num_qubits, num_actions):
        super(Actor, self).__init__()
        
        state_size = 2**num_qubits * 2  # Real and imaginary parts
        
        self.network = nn.Sequential(
            nn.Linear(state_size + 2, 256),  # +2 for circuit depth and reward
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.414)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Dictionary containing:
                - circuit_state: Tensor of shape (batch_size, 2^num_qubits, 2)
                - circuit_depth: Tensor of shape (batch_size, 1)
                - reward: Tensor of shape (batch_size, 1)
        
        Returns:
            action_probs: Tensor of shape (batch_size, num_actions)
        """
        # Flatten circuit state
        circuit_state = state['circuit_state'].view(state['circuit_state'].size(0), -1)
        
        x = torch.cat([
            circuit_state,
            state['circuit_depth'],
            state['reward']
        ], dim=1)
        
        logits = self.network(x)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy.
        
        Args:
            state: Current state
            deterministic: If True, return the most probable action
        
        Returns:
            action: Sampled action
            action_log_prob: Log probability of the action
        """
        action_probs = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            action_log_prob = torch.log(action_probs[torch.arange(action_probs.size(0)), action])
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob 