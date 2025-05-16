import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_city
import matplotlib.pyplot as plt

class QuantumCircuitEnv(gym.Env):
    """
    Custom Environment for quantum circuit optimization using RL.
    The environment allows an agent to rearrange quantum gates to optimize
    the circuit for solving the Max-Cut problem.
    """
    
    def __init__(self, num_qubits=4, max_steps=100):
        super(QuantumCircuitEnv, self).__init__()
        
        self.num_qubits = num_qubits
        self.max_steps = max_steps
        self.current_step = 0
        
        # n qubits: n*(n-1)/2 pairs
        num_possible_swaps = (num_qubits * (num_qubits - 1)) // 2
        self.action_space = spaces.Discrete(num_possible_swaps)
        
        # obs space
        # 1 Current circuit state (2^num_qubits complex numbers)
        # 2 Current circuit depth
        # 3 Current reward
        self.observation_space = spaces.Dict({
            'circuit_state': spaces.Box(
                low=-1, high=1,
                shape=(2**num_qubits, 2),  # Re and Im parts
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
        statevector = Statevector.from_instruction(self.circuit)
        state = statevector.data
        
        return {
            'circuit_state': np.stack([state.real, state.imag], axis=1),
            'circuit_depth': np.array([self.circuit.depth()]),
            'reward': np.array([self.current_reward])
        }
    
    def _calculate_reward(self):
        """
        Calculate reward form
        1 Solution quality (Max-Cut objective)
        2 Circuit depth (penalty for longer circuits)
        3 Gate count (penalty for more gates)
        """
        # TODO: Implement Max-Cut objective calculation
        # For now, return a simple reward based on circuit depth
        depth_penalty = -0.1 * self.circuit.depth()
        gate_penalty = -0.05 * len(self.circuit.data)
        
        return depth_penalty + gate_penalty
    
    def step(self, action):
        """
        in
            action: Integer representing which qubit pair to swap
            
        out
            observation: Current state of the circuit
            reward: Reward for the current state
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        self.current_step += 1
        
        qubit_pairs = [(i, j) for i in range(self.num_qubits) 
                      for j in range(i+1, self.num_qubits)]
        qubit1, qubit2 = qubit_pairs[action]
        
        self.circuit.swap(qubit1, qubit2)
        
        self.current_reward = self._calculate_reward()
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_observation(), self.current_reward, terminated, truncated, {}
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        qr = QuantumRegister(self.num_qubits, 'q')
        cr = ClassicalRegister(self.num_qubits, 'c')
        self.circuit = QuantumCircuit(qr, cr)
        
        for i in range(self.num_qubits):
            self.circuit.h(i)
        
        self.current_reward = self._calculate_reward()
        
        return self._get_observation(), {}
    
    def render(self):
        self.circuit.draw('mpl')
        plt.show() 