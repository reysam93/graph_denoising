import cvxpy as cp
import numpy as np
from scipy.linalg import khatri_rao


def filter_id(Y, X, S, gamma, delta, Cy):
    """
    Performs the filter identification step of the robust filter identification algorithm.
    It estimates H using cvx, not the analytical solution.
    """
    H = cp.Variable(S.shape, symmetric=True)

    ls_loss = cp.sum_squares(Y - H@X)
    commut_loss = cp.sum_squares(H@S - S@H)
    commut_cy_loss = cp.sum_squares(H@Cy - Cy@H)

    obj = ls_loss + gamma*commut_loss + delta*commut_cy_loss
    prob = cp.Problem(cp.Minimize(obj))
    try:
        prob.solve()
    except cp.SolverError:
        print("WARNING: Could not find optimal H -- Solver Error")
        return None

    if prob.status in ["optimal", "optimal_inaccurate"]:
            return H.value
    
    print(f"WARNING: problem status: {prob.status}")
    return None


def graph_id(Sn, H, Cy, lambd, gamma, delta):
    """
    Performs the filter identification step of the robust filter identification algorithm
    """
    S = cp.Variable(H.shape, symmetric=True)
    s_loss = cp.norm(S - Sn, 1)
    commut_loss = cp.sum_squares(H@S - S@H)
    commut_cy_loss = cp.sum_squares(S@Cy - Cy@S)
    # TODO: add sparsity loss

    obj = lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss
    constraints = [S >= 0, cp.diag(S) == 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve()
    except cp.SolverError:
        print("WARNING: Could not find optimal S -- Solver Error")
        return None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return S.value
    else:
        print(f"WARNING: problem status: {prob.status}")
        return None


def estH_iter(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4):
    lambd, gamma, delta, inc_gamma = params

    N, M = X.shape
    S_prev = Sn
    H_prev = Sn
    S = Sn

    err = []

    count_es = 0
    min_err = np.inf

    for i in range(max_iters):
        # Filter identification problem
        H = filter_id(Y, X, S, gamma, delta, Cy)
        H = H_prev if H is None else H

        # Graph identification
        S = graph_id(Sn, H, Cy, lambd, gamma, delta)
        S = S_prev if S is None else S

        # Check convergence
        ls_loss = ((Y - H@X)**2).sum()
        s_loss = np.abs(S-Sn).sum()
        commut_loss = ((S@H - H@S)**2).sum()
        commut_cy_loss = ((Cy@H - H@Cy)**2).sum()
        err.append(ls_loss + lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss)
        # print(f"Iter: {i} - err: {err[i]}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th:
            H_min = H
            S_min = S
            i_min = i
            print(f'\t\tConvergence reached at iteration {i}')
            break
        if err[i] > min_err:
            count_es += 1
        else:
            min_err = err[i]
            H_min = H.copy()
            S_min = S.copy()
            i_min = i
            count_es = 0
        
        if count_es == patience:
            break

        gamma = inc_gamma*gamma if inc_gamma else gamma
    
    return i_min, H_min, S_min


def estH_iter_rew(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4):
    lambd, gamma, delta, beta, inc_gamma = params

    N, M = X.shape
    S_prev = Sn
    H_prev = Sn
    S = Sn

    err = []

    W1 = np.ones((N,N))
    W2 = np.ones((N,N))
    delta1 = 1e-3
    delta2 = 1e-3

    count_es = 0
    min_err = np.inf

    for i in range(max_iters):
        # Filter identification problem
        H = filter_id(Y, X, S, gamma, delta, Cy)
        H = H_prev if H is None else H

        # Graph identification
        S = cp.Variable((N,N), symmetric=True)

        sn_loss = cp.sum(cp.multiply(W1, cp.abs(S - Sn)))
        s_loss = cp.sum(cp.multiply(W2, S))
        commut_loss = cp.sum_squares(H@S - S@H)
        commut_cy_loss = cp.sum_squares(S@Cy - Cy@S)

        obj = lambd*sn_loss + beta*s_loss + gamma*commut_loss + delta*commut_cy_loss

        constraints = [
            S >= 0,
            cp.diag(S) == 0
        ]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        try:
            prob.solve()
        except cp.SolverError:
            #print("estH_iter_rew -- Could not find optimal S -- Solver Error")
            S = S_prev
        except cp.DCPError:
            raise RuntimeError("estH_iter_rew -- Could not find optimal S -- DCP Error")

        if prob.status == "optimal":
            S = S.value
        else:
            #print("estH_iter_rew -- Could not find optimal S")
            S = S_prev
        
        W1 = lambd / (np.abs(S - Sn) + delta1)
        W2 = beta / (S + delta2)

        ls_loss = ((Y - H@X)**2).sum()
        s_loss = np.abs(S-Sn).sum()
        commut_loss = ((S@H - H@S)**2).sum()
        commut_cy_loss = ((Cy@H - H@Cy)**2).sum()
        err.append(ls_loss + lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss)
        # print(f"Iter: {i} - err: {err[i]}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th:
            H_min = H
            S_min = S
            i_min = i
            print(f'Convergence reached at iteration {i}')
            break
        if err[i] > min_err:
            count_es += 1
        else:
            min_err = err[i]
            H_min = H.copy()
            S_min = S.copy()
            i_min = i
            count_es = 0
        
        if count_es == patience:
            break
        if inc_gamma:
            gamma = inc_gamma*gamma
    
    return i_min, H_min, S_min

def estH_denS(X, Y, Sn, Cy, params):
    # import warnings
    # warnings.filterwarnings("ignore")
    N, M = X.shape

    gamma, delta = params

    # Denoise S
    S = cp.Variable((N,N), symmetric=True)

    s_loss = cp.norm(S - Sn, 1)
    commut_cy_loss = cp.sum_squares(S@Cy - Cy@S)

    obj = s_loss + delta*commut_cy_loss

    constraints = [
        S >= 0,
        cp.diag(S) == 0,
        #commut_cy_loss <= delta
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve()
    except cp.SolverError:
        #print("estH_denS -- Could not find optimal S -- Solver Error")
        S = Sn

    if prob.status == "optimal":
        S = S.value
    else:
        #print("estH_denS -- Could not find optimal S")
        S = Sn

    # Compute H from S
    H = cp.Variable((N,N), symmetric=True)

    ls_loss = cp.sum_squares(Y - H@X)

    commut_loss = cp.sum_squares(H@S - S@H)
    commut_cy_loss = cp.sum_squares(H@Cy - Cy@H)

    obj = ls_loss + gamma*commut_loss + delta*commut_cy_loss

    #const = [commut_loss <= 0]

    prob = cp.Problem(cp.Minimize(obj))#, const)
    prob.solve()

    if prob.status == "optimal":
        H = H.value
    else:
        print("estH_denS -- Could not find optimal H")
        H = np.zeros((N,N))

    ls_loss = ((Y - H@X)**2).sum()
    s_loss = np.abs(S-Sn).sum()
    commut_loss = ((S@H - H@S)**2).sum()
    commut_cy_loss = ((Cy@H - H@Cy)**2).sum()
    err = ls_loss + s_loss + gamma*commut_loss + delta*commut_cy_loss

    return -1, H, S

def estH_unpertS(X, Y, S, Cy=None, params=None):
    """
    Estimation of the graph filter coefficients (h) assuming the GSO S is known.
    It uses knowledge only of the eigenvectors of S. The eigenvalues are ignored. 
    """
    _, eigvecs = np.linalg.eigh(S)

    # Z is an MNxN matrix
    Z = khatri_rao(X.T @ eigvecs, eigvecs)
    h_freq, _, _, _ = np.linalg.lstsq(Z, Y.flatten('F'), rcond=None)
    H = eigvecs @ np.diag(h_freq) @ eigvecs.T
    return -1, H, S


def fi_eigval(X, Y, S, K):
    """
    Estimation of the graph filter coefficients (h) assuming the GSO S is known.
    It uses knowledge of both the eigenvectors and eigenvalues of S. 
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    Psi = np.vander(eigvals, K, increasing=True)

    # Z is an MNxL
    Z = khatri_rao(X.T @ eigvecs, eigvecs)@Psi
    h, _, _, _ = np.linalg.lstsq(Z, Y.flatten('F'), rcond=None)
    H = eigvecs @ np.diag(Psi@h) @ eigvecs.T
    return H, S, h


def rfi(X, Y, Sn, Cy, params, max_iters=20, th=1e-3):
    # Initialization
    lambd, gamma, delta, inc_gamma = params
    N = X.shape[0]
    S_prev = S = Sn

    # Precomputing quantities
    X_kron = np.kron(X@X.T, np.eye(N))
    Y_kron = np.kron(X, np.eye(N))@Y.flatten(order='F')

    diff_H = np.zeros(max_iters - 1)
    diff_S = np.zeros(max_iters - 1)
    for i in range(max_iters):
        # Filter identification step
        Z = np.kron(S@S, np.eye(N)) + np.kron(np.eye(N), S@S) - 2*np.kron(S, S)
        H = (np.linalg.inv(X_kron + gamma*Z)@Y_kron).reshape((N,N), order='F')

        # Graph identification
        S = graph_id(Sn, H, Cy, lambd, gamma, delta)
        S = S_prev if S is None else S

        gamma = inc_gamma*gamma if inc_gamma else gamma

        if i == 0:
            H_prev = H
            S_prev = S
            continue

        norm_H_prev = np.linalg.norm(H_prev, 'fro')
        norm_S_prev = np.linalg.norm(S_prev, 'fro')
        diff_H[i-1] = (np.linalg.norm(H - H_prev, 'fro')/norm_H_prev)**2
        diff_S[i-1] = (np.linalg.norm(S - S_prev, 'fro')/norm_S_prev)**2
        if diff_H[i-1] < th and diff_S[i-1] < th:
            print(f'Convergence reached at iteration {i}')
            return H, S , diff_H, diff_S

        H_prev = H
        S_prev = S
                    
    return H, S, diff_H, diff_S


# def efficient_filter_id(h_prev):
#     mu = 1




# def rfi_pgd(X, Y, Sn):
#     N = X.shape[0]

#     H = efficient_filter_id()



##########   DEBUG METHODS   ########## 
def rfi_debug(X, Y, Sn, Cy, params, H_true, S_true, max_iters=20):
    # Initialization
    lambd, gamma, delta, inc_gamma = params
    N, M = X.shape
    S_prev = S = Sn

    # Precomputing quantities
    norm_H = np.linalg.norm(H_true,'fro')
    norm_S = np.linalg.norm(S_true, 'fro')

    X_kron = np.kron(X@X.T, np.eye(N))
    Y_kron = np.kron(X, np.eye(N))@Y.flatten(order='F')

    err_S = np.zeros(max_iters)
    err_H = np.zeros(max_iters)
    for i in range(max_iters):
        # Filter identification step
        Z = np.kron(S@S, np.eye(N)) + np.kron(np.eye(N), S@S) - 2*np.kron(S, S)
        H = (np.linalg.inv(X_kron + gamma*Z)@Y_kron).reshape((N,N), order='F')

        # Graph identification
        S = graph_id(Sn, H, Cy, lambd, gamma, delta)
        if S is None:
            S = S_prev

        if inc_gamma:
            gamma = inc_gamma*gamma

        err_H[i] = (np.linalg.norm(H - H_true, 'fro')/norm_H)**2
        err_S[i] = (np.linalg.norm(S - S_true, 'fro')/norm_S)**2

    return H, S, err_H, err_S



