import torch
import torch.nn as nn

class Critic(nn.Module):
    
    def __init__(self, num_qubits):
        super(Critic, self).__init__()
        
        state_size = 2**num_qubits * 2  # Real and imaginary parts
        
        self.network = nn.Sequential(
            nn.Linear(state_size + 2, 256),  # +2 for circuit depth and reward
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.414)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        args:
            state: pydict
                - circuit_state: Tensor of shape (batch_size, 2^num_qubits, 2)
                - circuit_depth: Tensor of shape (batch_size, 1)
                - reward: Tensor of shape (batch_size, 1)
        outpuit
            value: Tensor of shape (batch_size, 1)
        """
        circuit_state = state['circuit_state'].view(state['circuit_state'].size(0), -1)
        
        x = torch.cat([
            circuit_state,
            state['circuit_depth'],
            state['reward']
        ], dim=1)
        
        value = self.network(x)
        
        return value 