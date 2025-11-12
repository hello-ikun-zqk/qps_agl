import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def sinkhorn_knopp(matrix, tol=1e-9, max_iter=1000):
    """Sinkhorn-Knopp algorithm to make a matrix doubly stochastic."""
    for _ in range(max_iter):
        # Normalize rows and columns alternately
        matrix /= matrix.sum(axis=1, keepdims=True)
        matrix /= matrix.sum(axis=0, keepdims=True)
        
        # Check convergence
        if np.allclose(matrix.sum(axis=1), 1, atol=tol) and \
           np.allclose(matrix.sum(axis=0), 1, atol=tol):
            break
    return matrix

def generate_minimum_spanning_tree(n):
    """Generate a minimum spanning tree as an adjacency matrix."""
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0, 1)  # Random edge weights

    mst = nx.minimum_spanning_tree(G, weight='weight')
    return nx.to_numpy_array(mst, nodelist=range(n))

def get_doubly_random_matrix_with_probabilities(n, p=0.5, diagonal="random"):
    """Generate a doubly stochastic matrix based on a random graph."""
    matrix = generate_minimum_spanning_tree(n)

    # Add edges based on probability p
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] == 0 and np.random.rand() <= p:
                matrix[i, j] = matrix[j, i] = 1

    if diagonal == "1":
        np.fill_diagonal(matrix, 1)
    elif diagonal == "0":
        np.fill_diagonal(matrix, 0)

    return sinkhorn_knopp(matrix)

def get_doubly_random_matrix(n):
    return get_doubly_random_matrix_with_probabilities(n,p=0.6,diagonal= "1")

def get_doubly_stochastic_star_network(n, diagonal="0"):
    """Generate a star network as a doubly stochastic matrix."""
    matrix = np.zeros((n, n))
    matrix[0, 1:] = matrix[1:, 0] = 1  # Connect center node to others

    if diagonal == "1":
        np.fill_diagonal(matrix, 1)
    elif diagonal == "0":
        np.fill_diagonal(matrix, 0)

    return sinkhorn_knopp(matrix)

def get_doubly_stochastic_ring_network(n, diagonal="0"):
    """Generate a ring network as a doubly stochastic matrix."""
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, (i - 1) % n] = matrix[i, (i + 1) % n] = 1  # Connect neighbors

    if diagonal == "1":
        np.fill_diagonal(matrix, 1)
    elif diagonal == "0":
        np.fill_diagonal(matrix, 0)

    return sinkhorn_knopp(matrix)

def get_strongly_connected_directed_row_column_stochastic_matrix(n, show=False):
    """Generate strongly connected row and column stochastic matrices."""
    while True:
        G = nx.fast_gnp_random_graph(n, p=0.5, directed=True)
        if nx.is_strongly_connected(G):
            adjacency_matrix = nx.to_numpy_array(G)
            np.fill_diagonal(adjacency_matrix, 1)
            
            if show:
                draw_directed_graph(adjacency_matrix)

            row_stochastic = adjacency_matrix / adjacency_matrix.sum(axis=1, keepdims=True)
            col_stochastic = adjacency_matrix / adjacency_matrix.sum(axis=0, keepdims=True)

            return row_stochastic, col_stochastic

def generate_strongly_connected_weight_matrix(n,show=False):
    """Generate a strongly connected directed graph with self-loops and construct A."""
    while True:
        # Generate a random directed graph
        G = nx.fast_gnp_random_graph(n, p=0.5, directed=True)
        
        # Ensure the graph is strongly connected
        if nx.is_strongly_connected(G):
            adjacency_matrix = nx.to_numpy_array(G)
            
            # Add self-loops
            np.fill_diagonal(adjacency_matrix, 1)

            if show:
                draw_directed_graph(adjacency_matrix)

            # Compute out-degree for each node
            out_degrees = adjacency_matrix.sum(axis=0)

            # Construct A matrix
            A = np.zeros_like(adjacency_matrix)
            for j in range(n):
                if out_degrees[j] > 0:
                    A[:, j] = adjacency_matrix[:, j] / out_degrees[j]

            return A

def draw_directed_graph(adj_matrix,layout=None):
    """Visualize a directed graph from its adjacency matrix."""
    G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    if layout is None or layout=="spring":
        pos = nx.spring_layout(G)
    elif layout=="circular":
        pos = nx.circular_layout(G)  # Arrange nodes in a circle
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, 
            arrowstyle='-|>', arrowsize=15, edge_color='gray')
    plt.show()

def draw_undirected_graph(adj_matrix,layout=None):
    """Visualize an undirected graph from its adjacency matrix."""
    G = nx.from_numpy_matrix(adj_matrix)
    if layout is None or layout=="spring":
        pos = nx.spring_layout(G)
    elif layout=="circular":
        pos = nx.circular_layout(G)  # Arrange nodes in a circle
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, 
            edge_color='gray', width=1.5)
    plt.show()


def verify_weight_matrix(A):
    """Verify if the weight matrix satisfies the given assumptions."""
    n = A.shape[0]

    # Check column stochasticity
    col_sums = A.sum(axis=0)
    is_col_stochastic = np.allclose(col_sums, 1, atol=1e-9)

    # Check positive diagonal entries
    diag_positive = np.all(np.diag(A) > 0)

    # Check strong connectivity
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    is_strongly_connected = nx.is_strongly_connected(G)

    return is_col_stochastic, diag_positive, is_strongly_connected

# Example usage
if __name__ == "__main__":
    n = 5  # Number of nodes

    # Generate and visualize a doubly stochastic random matrix
    # matrix = get_doubly_random_matrix_with_probabilities(n, p=0.5)
    # print(matrix)
    # draw_undirected_graph(matrix,"circular")
    matrix = generate_strongly_connected_weight_matrix(n)
    print(matrix)
    draw_directed_graph(matrix,"circular")
