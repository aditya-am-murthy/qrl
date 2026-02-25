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
    # Cirq Rz(t) = exp(-i Z t/2), so |1⟩ gets phase e^{i t/2}. We want e^{-i gamma} when edge is cut,
    # so use Rz(-2*gamma) so that |1⟩ gets e^{-i gamma}.
    for edge in graph_edges:
        i, j = edge
        circuit.append([
            cirq.CNOT(qubits[i], qubits[j]),
            cirq.rz(-2 * gamma).on(qubits[j]),
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
    n_qubits = int(np.log2(len(probs)))
    # Cirq state vector: default order is big-endian (qubit 0 = MSB).
    # So state index i has binary string where position 0 = qubit 0.
    expectation = 0
    for state_idx, prob in enumerate(probs):
        binary = format(state_idx, f'0{n_qubits}b')
        cut_value = 0
        for (a, b) in graph_edges:
            if binary[a] != binary[b]:
                cut_value += 1
        expectation += prob * cut_value
    return expectation

def maxcut_expectation_from_circuit(circuit, graph_edges, simulator=None):
    """
    Compute Max-Cut expectation (expected number of edges cut) for a circuit's final state.
    """
    if simulator is None:
        simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    return calculate_maxcut_objective(result, graph_edges)


def cut_value_of_bitstring(state_index, n_qubits, graph_edges):
    """Cut size if we interpret state_index as a partition (qubit k = bit k in big-endian)."""
    binary = format(state_index, f'0{n_qubits}b')
    return sum(1 for (a, b) in graph_edges if binary[a] != binary[b])


def sample_cuts_from_state(statevector_result, graph_edges, num_samples, rng=None):
    """
    Sample num_samples bitstrings from the state's distribution; return cut for each.
    Returns (mean cut over samples, best cut over samples).
    Use this to compare quantum 'best solution in K shots' vs classical single-solution.
    """
    if rng is None:
        rng = np.random.default_rng()
    probs = np.abs(statevector_result.final_state_vector) ** 2
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    n_qubits = int(np.log2(len(probs)))
    indices = rng.choice(len(probs), size=num_samples, p=probs)
    cuts = [cut_value_of_bitstring(int(idx), n_qubits, graph_edges) for idx in indices]
    return np.mean(cuts), max(cuts)


def expectation_and_best_of_samples(circuit, graph_edges, num_samples=100, simulator=None, rng=None):
    """
    Simulate circuit once; return (expectation, best cut among num_samples).
    Lets you compare 'best solution in K shots' (quantum) vs 'one run' (e.g. local search).
    """
    if simulator is None:
        simulator = cirq.Simulator()
    if rng is None:
        rng = np.random.default_rng()
    result = simulator.simulate(circuit)
    expectation = calculate_maxcut_objective(result, graph_edges)
    _, best = sample_cuts_from_state(result, graph_edges, num_samples, rng)
    return expectation, best


def greedy_maxcut(num_vertices, graph_edges):
    """
    Greedy Max-Cut: assign each vertex to the partition that increases the cut.
    Returns the cut size (number of edges between partitions).
    """
    partition = {}  # vertex -> 0 or 1
    for v in range(num_vertices):
        cut_if_0 = sum(1 for (a, b) in graph_edges if (a == v and partition.get(b, 0) == 1) or (b == v and partition.get(a, 0) == 1))
        cut_if_1 = sum(1 for (a, b) in graph_edges if (a == v and partition.get(b, 0) == 0) or (b == v and partition.get(a, 0) == 0))
        partition[v] = 0 if cut_if_0 >= cut_if_1 else 1
    return sum(1 for (a, b) in graph_edges if partition[a] != partition[b])


def local_search_maxcut(num_vertices, graph_edges, max_iters=1000):
    """
    Local search (flip vertex to other partition if it improves cut).
    Starts from greedy solution. Returns cut size.
    """
    partition = {}  # vertex -> 0 or 1
    for v in range(num_vertices):
        cut_if_0 = sum(1 for (a, b) in graph_edges if (a == v and partition.get(b, 0) == 1) or (b == v and partition.get(a, 0) == 1))
        cut_if_1 = sum(1 for (a, b) in graph_edges if (a == v and partition.get(b, 0) == 0) or (b == v and partition.get(a, 0) == 0))
        partition[v] = 0 if cut_if_0 >= cut_if_1 else 1

    def cut_size():
        return sum(1 for (a, b) in graph_edges if partition[a] != partition[b])

    for _ in range(max_iters):
        improved = False
        for v in range(num_vertices):
            current = cut_size()
            partition[v] = 1 - partition[v]
            if cut_size() > current:
                improved = True
                break
            partition[v] = 1 - partition[v]
        if not improved:
            break
    return cut_size()


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