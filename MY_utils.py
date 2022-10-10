import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx import is_connected, to_numpy_array

def gen_data(N, M, p_er, p_n, eps, K = 4):
    #Generating graph and graph filter
    for _ in range(100): # 100 tries to make sure the graph is connected
        G = erdos_renyi_graph(N, p_er)
        if is_connected(G):
            break
    if not is_connected(G):
        raise RuntimeError("Could not create connected graph")
    W = to_numpy_array(G)

    # Generate graph filter
    Spow = np.eye(N)
    H = np.zeros((N,N))
    h = 2 * np.random.rand(K) - 1
    h = h / np.linalg.norm(h)
    for k in range(K):
        H += h[k] * Spow
        Spow = Spow @ W

    # TODO: we should consider different types of perturbations. This one may result in denser perturbed graphs
    # Perturbated adjacency
    adj_pert_idx = np.triu(np.random.rand(N,N) < eps, 1)
    adj_pert_idx = adj_pert_idx + adj_pert_idx.T
    Wn = np.logical_xor(W, adj_pert_idx).astype(float)


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

    return X, Y, Cy, Cy_samp, H, W, Wn, h


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