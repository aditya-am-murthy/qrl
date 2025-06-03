import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cirq
from cirq import ops
import matplotlib.pyplot as plt

from utils.circuit_utils import (
    create_maxcut_circuit,
    calculate_maxcut_objective,
    generate_random_graph
)

class QuantumCircuitEnv(gym.Env):
    """
    Custom Environment for quantum circuit optimization using RL.
    The environment allows an agent to rearrange quantum gates to optimize
    the circuit for solving the Max-Cut problem using Cirq.
    """
    
    def __init__(self, num_qubits=4, max_steps=100, edge_probability=0.5):
        super(QuantumCircuitEnv, self).__init__()
        
        self.num_qubits = num_qubits
        self.max_steps = max_steps
        self.current_step = 0
        self.edge_probability = edge_probability
        
        # Generate random graph for Max-Cut
        self.graph_edges = generate_random_graph(num_qubits, edge_probability)
        
        # Initialize QAOA parameters
        self.params = [np.pi/4, np.pi/4]  # Initial gamma and beta
        
        # Define action space (swap gates between positions)
        num_possible_swaps = (num_qubits * (num_qubits - 1)) // 2
        self.action_space = spaces.Discrete(num_possible_swaps)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'circuit_state': spaces.Box(
                low=-1, high=1,
                shape=(2**num_qubits, 2),  # Real and imaginary parts
                dtype=np.float32
            ),
            'circuit_depth': spaces.Box(
                low=0, high=float('inf'),
                shape=(1,),
                dtype=np.float32
            ),
            'reward': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(1,),
                dtype=np.float32
            )
        })
        
        self.reset()
    
    def _get_observation(self):
        """Convert current circuit state to observation."""
        simulator = cirq.Simulator()
        result = simulator.simulate(self.circuit)
        state = result.final_state_vector
        
        return {
            'circuit_state': np.stack([state.real, state.imag], axis=1),
            'circuit_depth': np.array([len(self.circuit)]),  # Cirq uses circuit length as depth
            'reward': np.array([self.current_reward])
        }
    
    def _calculate_reward(self):
        """
        Calculate reward based on:
        1. Solution quality (Max-Cut objective)
        2. Circuit depth (penalty for longer circuits)
        3. Gate count (penalty for more gates)
        4. Improvement over previous state
        """
        # Get current statevector
        simulator = cirq.Simulator()
        result = simulator.simulate(self.circuit)
        
        # Calculate Max-Cut objective
        maxcut_value = calculate_maxcut_objective(result, self.graph_edges)
        
        # Calculate penalties with reduced weights
        depth_penalty = -0.01 * len(self.circuit)  # Reduced from -0.1
        gate_penalty = -0.005 * len(list(self.circuit.all_operations()))  # Reduced from -0.05
        
        # Calculate improvement over previous state
        improvement = 0
        if hasattr(self, 'previous_maxcut_value'):
            improvement = max(0, maxcut_value - self.previous_maxcut_value) * 2.0  # Bonus for improvement
        
        # Store current maxcut value for next step
        self.previous_maxcut_value = maxcut_value
        
        # Combine rewards with better scaling
        reward = (
            maxcut_value * 2.0 +  # Increase weight of MaxCut objective
            depth_penalty +
            gate_penalty +
            improvement  # Add improvement bonus
        )
        
        # Add small constant reward for each step to encourage exploration
        reward += 0.1
        
        return reward
    
    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: Integer representing which qubit pair to swap
            
        Returns:
            observation: Current state of the circuit
            reward: Reward for the current state
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        self.current_step += 1
        
        # Convert action to qubit pair
        qubit_pairs = [(i, j) for i in range(self.num_qubits) 
                      for j in range(i+1, self.num_qubits)]
        qubit1, qubit2 = qubit_pairs[action]
        
        # Apply swap gate
        self.circuit.append(cirq.SWAP(self.qubits[qubit1], self.qubits[qubit2]))
        
        # Calculate reward
        self.current_reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_observation(), self.current_reward, terminated, truncated, {}
    
    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Create qubits
        self.qubits = [cirq.LineQubit(i) for i in range(self.num_qubits)]
        
        # Create initial QAOA circuit
        self.circuit = create_maxcut_circuit(
            self.num_qubits,
            self.graph_edges,
            self.params
        )
        
        # Initialize previous maxcut value
        simulator = cirq.Simulator()
        result = simulator.simulate(self.circuit)
        self.previous_maxcut_value = calculate_maxcut_objective(result, self.graph_edges)
        
        self.current_reward = self._calculate_reward()
        
        return self._get_observation(), {}
    
    def render(self):
        """Render the current circuit state."""
        print(self.circuit)
        plt.figure(figsize=(10, 5))
        cirq.plot_state_histogram(self.circuit)
        plt.show() 