import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx import is_connected, to_numpy_array

# Graph Type Constants
SBM = 1
ER = 2
BA = 3

MAX_RETRIES = 20

def obtain_filter_coefs(S, H, K):
    """
    Obtain the coefficients h used to generate a graph filter H, using the GSO
    given as an argument
    """
    Svals, Svecs = np.linalg.eigh(S)
    Psi = np.fliplr(np.vander(Svals))
    Psi_inv = np.linalg.pinv(Psi[:,:K])

    h_bar = Svecs.T @ H @ Svecs
    h = Psi_inv @ np.diag(h_bar)

    return h

def generate_graph(N, g_params, seed=None):
    """
    Return a graph generated according to the parameters inside g_params dictionary.

    The values it must contain vary according to the type of graph to be created:
    * Erdos-Renyi Graph:
      * p: connection probability between nodes
    * Stochastic-Block Model:
      * p: connection probability between nodes of the same community
      * q: connection probability between nodes of different communities
    * Barabasi Albert:
      * m: Number of edges to attach from a new node to existing nodes

    Parameters:
        N: number of nodes of the graph
        g_params: parameters for the graph type
        seed: random number generator state
    """
    if g_params['type'] == ER:
        for _ in range(MAX_RETRIES): # MAX_RETRIES to make sure the graph is connected
            G = erdos_renyi_graph(N, g_params['p'])
            if is_connected(G):
                break
        if not is_connected(G):
            raise RuntimeError("Could not create connected graph")
    else:
        raise NotImplementedError("Only: ER graph available")
    return to_numpy_array(G)

def generate_graph_filter(S, K, neg_coefs=True, sort_h=False, norm_S=False, norm_H=False, return_h=False):
    """
    Generate a graph filter from the graph shift operator and random coefficients
    """
    # Generate graph filter
    if neg_coefs:
        h = 2 * np.random.rand(K) - 1
    else:
        h = np.random.rand(K)
    if sort_h:
        h = sorted(h, key=lambda x: np.abs(x))[::-1]
    h = h / np.linalg.norm(h)
    
    eigvals_S, eigvecs_S = np.linalg.eigh(S)
    if norm_S:
        eigvals_S = eigvals_S / np.max(np.abs(eigvals_S))
    psi = np.fliplr(np.vander(eigvals_S, K))

    eigvals_H = psi @ h
    H = eigvecs_S @ np.diag(eigvals_H) @ eigvecs_S.T

    if norm_H:
        norm_h = np.sqrt((H**2).sum())
        H = H / norm_h
    
    if return_h:
        return H, h
    else:
        return H

def pert_S(S, type="rewire", eps=0.1):
    """
    Perturbate a given graph shift operator/adjacency matrix

    There are two types of perturbation
    * prob: changes a value in the adjacency matrix with a certain
    probability. May result in denser graphs
    * rewire: rewire a percentage of original edges randomly
    """
    N = S.shape[0]

    if type == "prob":
        # Perturbated adjacency
        adj_pert_idx = np.triu(np.random.rand(N,N) < eps, 1)
        adj_pert_idx = adj_pert_idx + adj_pert_idx.T
        Sn = np.logical_xor(S, adj_pert_idx).astype(float)
    elif type == "rewire":
        # Edge rewiring
        idx_edges = np.where(np.triu(S) != 0)
        Ne = idx_edges[0].size
        unpert_edges = np.arange(Ne)
        for i in range(int(Ne*eps)):
            idx_modify = np.random.choice(unpert_edges)
             # To prevent modifying the same edge twice
            unpert_edges = np.delete(unpert_edges, np.where(unpert_edges == idx_modify))
            start = idx_edges[0][idx_modify]
            new_end = np.random.choice(np.delete(np.arange(N), start))
            idx_edges[0][idx_modify] = min(start, new_end)
            idx_edges[1][idx_modify] = max(start, new_end)
        Sn = np.zeros((N,N))
        Sn[idx_edges] = 1.
        assert np.all(np.tril(Sn) == 0)
        Sn = Sn + Sn.T
    else:
        raise NotImplementedError("Choose either prob or rewire types")

    return Sn

def gen_data(N, M, p_er, p_n, eps, K = 4, neg_coefs=True, sort_h=False, norm_S=False, norm_H=False, pert_type="rewire", seed=None):
    #Generating graph and graph filter
    S = generate_graph(N, {'type': ER, 'p': p_er}, seed=seed)

    # Generate graph filter
    H, h = generate_graph_filter(S, K, neg_coefs, sort_h, norm_S, norm_H, True)

    # Perturbate adjacency
    Sn = pert_S(S, pert_type, eps)

    # Generating data samples
    X = np.random.randn(N,M) / np.sqrt(N)
    Y = H@X
    norm_y = (Y**2).sum() / M
    Cy = H @ H.T

    # Adding noise
    Y += np.random.randn(N,M) * np.sqrt(norm_y*p_n / N)

    # Sample covariance
    Cy_samp = Y @ Y.T / M
    Cy_samp = Cy_samp / np.sqrt((Cy_samp**2).sum())

    return X, Y, Cy, Cy_samp, H, S, Sn, h


def gen_data_sev_H(N, M, T, p_er, p_n, eps, K = 4):
    #Generating graph and graph filter
    for _ in range(100):
        G = erdos_renyi_graph(N, p_er)
        if is_connected(G):
            break
    if not is_connected(G):
        raise RuntimeError("Could not create connected graph")
    W = to_numpy_array(G)

    # Generate graph filter
    Spow = np.array([np.linalg.matrix_power(W, k) for k in range(K)])
    Hs = np.zeros((T, N, N))
    for i in range(T):
        h = 2*np.random.rand(K)-1
        h = h / np.linalg.norm(h)
        Hs[i,:,:] = sum(h[k] * Spow[k] for k in range(K))

    #Perturbated adjacency
    adj_pert_idx = np.triu(np.random.rand(N,N) < eps, 1)
    adj_pert_idx = adj_pert_idx + adj_pert_idx.T
    Wn = np.logical_xor(W, adj_pert_idx).astype(float)

    # Generating data samples
    X = np.random.randn(T, N, M) / np.sqrt(N)
    Y = Hs@X
    Cy = Hs @ Hs.transpose(0,2,1)

    # Adding noise
    norm_y = (Y**2).sum((1,2)) / M
    Y += np.random.randn(T,N,M) * np.sqrt(norm_y*p_n / N)[:,None,None]

    # Sample covariance
    Cy_samp = Y @ Y.transpose(0,2,1)
    Cy_samp = Cy_samp / (np.sqrt((Cy_samp**2).sum((1,2)))[:,None,None])

    return X, Y, Cy, Cy_samp, Hs, W, Wn