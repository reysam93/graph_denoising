import numpy as np
from scipy.sparse import kronsum


def gradient_filter_id(XX, S, y_kron, h, gamma):
    """
    Computes the gradient of the objective funciton employed in the filter identification step.
    """
    N = XX.shape[0]

    # Define vectorization for  kron(I, A)b  operation and block-diag times a vector
    diag_blocks_vec = lambda A, b: np.array([A[:,i].repeat(N)*np.tile(b[i*N:(i+1)*N], N) for i in range(N)]).sum(axis=0)
    block_diag_vec = lambda A, b: np.concatenate([A@b[i*N:(i+1)*N] for i in range(N)])

    zh = block_diag_vec(S, h) - diag_blocks_vec(S, h)
    zzh = block_diag_vec(S, zh) - diag_blocks_vec(S, zh)
    return diag_blocks_vec(XX, h) - y_kron + gamma*zzh


def efficient_filter_id(XX, S, y_kron, gamma, mu, H_init=None, max_iters=100, eps=1e-3, verbose=False):
    """
    Solves the filter identification step following a gradient descent algorithm
    """
    h_iters = np.zeros((max_iters, y_kron.shape[0]))
    h = H_init.flatten('F') if H_init is not None else np.zeros(y_kron.shape[0])

    for i in range(max_iters):
        h = h - mu*gradient_filter_id(XX, S, y_kron, h, gamma)
        h_iters[i,:] = h

        if verbose:
            print('Iter', i, ': norm(h):', np.linalg.norm(h))

        # Stopping condition    
        if i > 0:
            if np.linalg.norm(h_iters[i,:] - h_iters[i-1,:])/np.linalg.norm(h_iters[i-1,:]) < eps:
                break

    return h.reshape(S.shape, order='F'), h_iters


def sparse_graph_update(Z_T, s, sn, r, lambd, beta, gamma):
    for i, (s_i, sn_i, z_i) in enumerate(zip(s, sn, Z_T)): 
        r_i = np.asarray(r - z_i*s_i).squeeze() if s_i != 0 else r
        z_times_r = (z_i@r_i)[0]
        z_times_z = z_i.power(2).sum()

        ## get s[i] with projected soft-thresholding
        if (-lambd - beta - gamma*z_times_r)/(gamma*z_times_z) > sn_i:
            s_i = (-lambd - beta - gamma*z_times_r)/(gamma*z_times_z)

        elif (lambd - beta - gamma*z_times_r)/(gamma*z_times_z) < sn_i:
            s_i = (lambd - beta - gamma*z_times_r)/(gamma*z_times_z)

        else:
            s_i = sn_i

        s[i] = s_i if s_i > 0 else 0
        r = np.asarray(r_i + z_i*s_i).squeeze() if s_i != 0 else r_i

    return s, r


def efficient_graph_id(Sn, H, lambd, beta, gamma, S_init=None, max_iters=50, eps=5e-3, verbose=False):
    """
    Solves the graph denosing algorithm following a cyclic coordinate descent
    """
    N = Sn.shape[0]
    low_idxs = np.concatenate([np.arange(i+1, N)+N*i for i in range(N)])
    up_idxs = np.concatenate([np.arange((i+1)*N, N*(N-1)+1, N)+i for i in range(0,N-1)])
    sn = Sn.flatten('F')[low_idxs]

    # Assuming the graph is undirected
    Z_T_whole = kronsum(-H, H, format='csr')
    Z_T = Z_T_whole[low_idxs,:] + Z_T_whole[up_idxs,:]
    if S_init is not None:
        s = S_init.flatten('F')[low_idxs]
        r = s@Z_T
    else:
        s = np.zeros(sn.size)
        r = np.zeros(Z_T.shape[1])

    s_iters = np.zeros((max_iters, sn.size))
    for i in range(max_iters):
        s, r = sparse_graph_update(Z_T, s, sn, r, lambd, beta, gamma)
        s_iters[i,:] = s

        if verbose:
            print('Iter', i, ': norm(s):', np.linalg.norm(s))

        # Stopping condition    
        if i > 0:
            if np.linalg.norm(s_iters[i,:] - s_iters[i-1,:])/np.linalg.norm(s_iters[i-1,:]) < eps:
                break

    # reconstruction from triangular matrix
    S_flat = np.zeros(H.shape).flatten('F')
    mask_range = np.arange(N)
    mask_lower = mask_range[:,None] > mask_range
    S_flat[mask_lower.flatten('F')] = s
    S = S_flat.reshape((N,N), order='F')
    S = S + S.T
    return S, s_iters


def efficient_rfi(X, Y, Sn, regs, iters_out=20, iters_filter=10, iters_graph=10, eps=1e-3):
    # Initialization
    lamb, gamma, beta, inc_gamma, mu = regs
    N = X.shape[0]

    # Precomputing quantities
    ## Use sparse functions?
    y_kron = np.kron(X, np.eye(N))@Y.flatten(order='F')
    XX = X@X.T
    
    # Init S and H to 0
    Ss_hat = np.zeros((iters_out+1, N, N))
    Hs_hat = np.zeros((iters_out+1, N, N))
    for i in range(1, iters_out+1):
        # Init as 0
        Hs_hat[i,:,:], _ = efficient_filter_id(XX, Ss_hat[i-1], y_kron, gamma, mu,
                                           max_iters=iters_filter)
        Ss_hat[i,:,:], _ = efficient_graph_id(Sn, Hs_hat[i], lamb, beta, gamma,
                                          max_iters=iters_graph)
        
        gamma *= inc_gamma

        if i == 0:
            continue

        # Stopping condition
        norm_H_prev = np.linalg.norm(Hs_hat[i-1], 'fro') if not np.all(Hs_hat[i-1] == 0) else 1
        norm_S_prev = np.linalg.norm(Ss_hat[i-1], 'fro') if not np.all(Ss_hat[i-1] == 0) else 1
        diff_H = (np.linalg.norm(Hs_hat[i] - Hs_hat[i-1], 'fro')/norm_H_prev)**2
        diff_S = (np.linalg.norm(Ss_hat[i] - Ss_hat[i-1], 'fro')/norm_S_prev)**2

        # print(f'Iter: {i}  -  Conv H: {diff_H:.6f}  -  Conv S: {diff_S:.6f}')

        if diff_H < eps and diff_S < eps:
            print(f'Convergence reached at iteration {i}')
            return Hs_hat[i], Ss_hat[i], Hs_hat, Ss_hat

    return Hs_hat[i], Ss_hat[i], Hs_hat, Ss_hat
