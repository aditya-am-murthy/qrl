import numpy as np
import cirq
from cirq import ops

def create_maxcut_circuit(num_qubits, graph_edges, params):
    """
    Create a QAOA circuit for Max-Cut problem using Cirq.
    
    Args:
        num_qubits: Number of qubits (vertices in the graph)
        graph_edges: List of tuples (i,j) representing edges in the graph
        params: List of parameters [gamma, beta] for QAOA
    
    Returns:
        cirq.Circuit: QAOA circuit for Max-Cut
    """
    # Create qubits
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    circuit = cirq.Circuit()
    
    # Initially equal superposition
    circuit.append(cirq.H.on_each(*qubits))
    
    gamma = params[0]
    for edge in graph_edges:
        i, j = edge
        circuit.append([
            cirq.CNOT(qubits[i], qubits[j]),
            cirq.rz(2 * gamma).on(qubits[j]),
            cirq.CNOT(qubits[i], qubits[j])
        ])
    
    beta = params[1]
    circuit.append(cirq.rx(2 * beta).on_each(*qubits))
    
    return circuit

def calculate_maxcut_objective(statevector, graph_edges):
    """
    Calculate the Max-Cut objective value for a given quantum state.
    
    Args:
        statevector: Quantum state vector (cirq.StateVectorTrialResult)
        graph_edges: List of tuples (i,j) representing edges in the graph
    
    Returns:
        float: Expected value of the Max-Cut objective
    """
    # Get probabilities from state vector
    probs = np.abs(statevector.final_state_vector) ** 2
    
    # Calculate expectation value
    expectation = 0
    for i, prob in enumerate(probs):
        # Convert index to binary string
        binary = format(i, f'0{int(np.log2(len(probs)))}b')
        
        # Calculate cut value for this basis state
        cut_value = 0
        for edge in graph_edges:
            i, j = edge
            if binary[i] != binary[j]:  # vertices in different partitions
                cut_value += 1
        
        expectation += prob * cut_value
    
    return expectation

def generate_random_graph(num_vertices, edge_probability=0.5):
    """
    Generate a random graph for Max-Cut problem.
    
    Args:
        num_vertices: Number of vertices in the graph
        edge_probability: Probability of an edge between any two vertices
    
    Returns:
        list: List of tuples (i,j) representing edges in the graph
    """
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if np.random.random() < edge_probability:
                edges.append((i, j))
    return edges 