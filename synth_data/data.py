import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx import is_connected, to_numpy_array, connected_watts_strogatz_graph, karate_club_graph

# Graph Type Constants
SBM = 1
ER = 2
BA = 3
SW = 4
ZACHARY = 5

MAX_RETRIES = 100

def obtain_filter_coefs(S, H, K, return_h_bar=False, use_H_vals=False):
    """
    Obtain the coefficients h used to generate a graph filter H, using the GSO
    given as an argument
    """
    Svals, Svecs = np.linalg.eigh(S)
    Psi = np.fliplr(np.vander(Svals))
    Psi_inv = np.linalg.pinv(Psi[:,:K])

    if use_H_vals:
        h_bar = np.linalg.eigvalsh(H)[::-1]
    else:
        h_bar = np.diag(Svecs.T @ H @ Svecs)
    h = Psi_inv @ h_bar

    if return_h_bar:
        return h, h_bar
    else:
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
    elif g_params['type'] == SW:
        G = connected_watts_strogatz_graph(N, g_params['k'], g_params['p'], MAX_RETRIES)
    elif g_params['type'] == ZACHARY:
        G = karate_club_graph()
    else:
        raise NotImplementedError("Only: ER graph available")
    return to_numpy_array(G)

def generate_graph_filter(S, K, neg_coefs=True, exp_coefs=False, coef=1, sort_h=False, norm_S=False, norm_H=False, return_h=False):
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
    if exp_coefs:
        h = [h[i]*np.exp(-i*coef) for i in range(K)]
    h = h / np.linalg.norm(h)
    #print(h)
    
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

def pert_S(S, type="rewire", eps=0.1, creat=None, dest=None, sel_ratio=1, sel_node_idx=0):
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
    elif type == "creat-dest":

        creat = creat if creat is not None else eps
        dest = dest if dest is not None else eps

        A_x_triu = S.copy()
        A_x_triu[np.tril_indices(N)] = -1

        no_link_i = np.where(A_x_triu == 0)
        link_i = np.where(A_x_triu == 1)
        Ne = link_i[0].size

        # Create links
        if sel_ratio > 1 and sel_node_idx > 0:
            ps = np.array([sel_ratio if no_link_i[0][i] < sel_node_idx or no_link_i[1][i] < sel_node_idx else 1 for i in range(no_link_i[0].size)])
            ps = ps / ps.sum()
        else:
            ps = np.ones(no_link_i[0].size) / no_link_i[0].size
        links_c = np.random.choice(no_link_i[0].size, int(Ne * creat),
                                replace=False, p=ps)
        idx_c = (no_link_i[0][links_c], no_link_i[1][links_c])

        # Destroy links
        if sel_ratio > 1 and sel_node_idx > 0:
            ps = np.array([sel_ratio if link_i[0][i] < sel_node_idx or link_i[1][i] < sel_node_idx else 1 for i in range(link_i[0].size)])
            ps = ps / ps.sum()
        else:
            ps = np.ones(link_i[0].size) / link_i[0].size
        links_d = np.random.choice(link_i[0].size, int(Ne * dest),
                                replace=False, p=ps)
        idx_d = (link_i[0][links_d], link_i[1][links_d])

        A_x_triu[np.tril_indices(N)] = 0
        A_x_triu[idx_c] = 1.
        A_x_triu[idx_d] = 0.
        Sn = A_x_triu + A_x_triu.T
    else:
        raise NotImplementedError("Choose either prob, rewire or creat-dest perturbation types")
    return Sn

def gen_data(N, M, g_params, p_n, eps, K = 4, neg_coefs=True, exp_coefs=False, coef=1, sort_h=False, norm_S=False, norm_H=False, pert_type="rewire", creat=None, dest=None, sel_ratio=1, sel_node_idx=0, seed=None):
    #Generating graph and graph filter
    S = generate_graph(N, g_params, seed=seed)
    N = S.shape[0] # When using Zachary's karate club, N is ignored, thus setting it here again

    # Generate graph filter
    H, h = generate_graph_filter(S, K, neg_coefs, exp_coefs, coef, sort_h, norm_S, norm_H, True)

    # Perturbate adjacency
    Sn = pert_S(S, pert_type, eps, creat, dest, sel_ratio, sel_node_idx)

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

def gen_data_SEM(N, M, g_params, p_n, eps, K = 4, neg_coefs=True, exp_coefs=False, coef=1., sort_h=False, norm_S=False, norm_H=False, pert_type="rewire", creat=None, dest=None, seed=None):
    np.random.seed(seed)
    X, Y_poly, Cy_poly, Cy_samp_poly, H_poly, S, Sn, h = gen_data(N, M, g_params, p_n, eps, K, neg_coefs, exp_coefs, coef, sort_h, norm_S, norm_H, pert_type, creat, dest, seed)

    H_sem = np.linalg.pinv(np.eye(N) - S)
    Y_sem = H_sem @ X
    norm_y = (Y_sem**2).sum() / M
    Cy_sem = H_sem @ H_sem.T

    # Adding noise
    Y_sem += np.random.randn(N,M) * np.sqrt(norm_y*p_n / N)

    # Sample covariance
    Cy_samp_sem = Y_sem @ Y_sem.T / M
    Cy_samp_sem = Cy_samp_sem / np.sqrt((Cy_samp_sem**2).sum())

    return X, Y_poly, Y_sem, Cy_poly, Cy_sem, Cy_samp_poly, Cy_samp_sem, H_poly, H_sem, S, Sn, h


def gen_data_sev_H(N, M, T, g_params, p_n, eps, K = 4, neg_coefs=True, exp_coefs=False, coef=1, sort_h=False, norm_S=False, norm_H=False, pert_type="rewire", creat=None, dest=None, seed=None):
    #Generating graph and graph filter
    S = generate_graph(N, g_params, seed=seed)
    N = S.shape[0] # When using Zachary's karate club, N is ignored, thus setting it here again
    #np.random.seed(seed)

    # Generate graph filter
    Hs = np.zeros((T, N, N))
    hs = np.zeros((T, K))
    for i in range(T):
        # Generate graph filters
        Hs[i,:,:], hs[i,:] = generate_graph_filter(S, K, neg_coefs, exp_coefs, coef, sort_h, norm_S, norm_H, True)

    # Perturbate adjacency
    Sn = pert_S(S, pert_type, eps, creat, dest)

    # Generating data samples
    X = np.random.randn(T, N, M) / np.sqrt(N)
    Y = Hs@X
    Cy = Hs @ Hs.transpose(0,2,1)

    # Adding noise
    norm_y = (Y**2).sum((1,2)) / M
    Y += np.random.randn(T,N,M) * np.sqrt(norm_y*p_n / N)[:,None,None]

    # Sample covariance
    Cy_samp = Y @ Y.transpose(0,2,1) / M
    Cy_samp = Cy_samp / (np.sqrt((Cy_samp**2).sum((1,2)))[:,None,None])

    return X, Y, Cy, Cy_samp, Hs, S, Sn, hs