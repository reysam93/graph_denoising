import cvxpy as cp
import numpy as np
from scipy.linalg import khatri_rao

# Imported here to be able to call getattr(opt, func_name)
from opt_lls import estH_llsscp

VERB = False

def filter_id(Y, X, S, gamma, delta, Cy, verb=VERB):
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
        if verb:
            print("WARNING: Could not find optimal H -- Solver Error")
        try:
            prob.solve(verbose=False)
            if verb:
                print("Solver error fixed")
        except:
            if verb:
                print("A second solver error")
            return None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return H.value
    
    if verb:
        print(f"WARNING: problem status: {prob.status}")
    return None


def graph_id(Sn, H, Cy, lambd, gamma, delta, verb=VERB):
    """
    Performs the filter identification step of the robust filter identification algorithm
    """
    S = cp.Variable(H.shape, symmetric=True)
    s_loss = cp.sum(cp.abs(S - Sn))
    commut_loss = cp.sum_squares(H@S - S@H)
    commut_cy_loss = cp.sum_squares(S@Cy - Cy@S)
    # TODO: add sparsity loss

    obj = lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss
    constraints = [S >= 0, cp.diag(S) == 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve()
    except cp.SolverError:
        if verb:
            print("WARNING: Could not find optimal S -- Solver Error")
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if verb:
                print("Solver error fixed")
        except cp.SolverError as e:
            if verb:
                print("A second solver error")
                print(e)
            return None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return S.value
    else:
        if verb:
            print(f"WARNING: problem status: {prob.status}")
        return None

def graph_id_rew(Sn, H, Cy, W1, W2, lambd, gamma, delta, beta, verb=VERB):
    """
    Performs the filter identification step of the robust filter identification algorithm
    with the reweighted alternative
    """
    N = Sn.shape[0]
    S = cp.Variable((N,N), symmetric=True)

    sn_loss = cp.sum(cp.multiply(W1, cp.abs(S - Sn)))
    s_loss = cp.sum(cp.multiply(W2, cp.abs(S)))
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
        if verb:
            print("WARNING: Could not find optimal S -- Solver Error")
        try:
            prob.solve(verbose=False)
            if verb:
                print("Solver error fixed")
        except:
            if verb:
                print("A second solver error")
            return None
    except cp.DCPError:
        raise RuntimeError("Could not find optimal S -- DCP Error")

    if prob.status in ["optimal", "optimal_inaccurate"]:
        S = S.value
    else:
        if verb:
            print(f"WARNING: problem status: {prob.status}")
        return None
    
    return S

def estH_iter(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4, H_true=None, S_true=None):
    import warnings
    warnings.filterwarnings("ignore")
    lambd, gamma, delta, inc_gamma = params

    N, M = X.shape
    S_prev = Sn
    H_prev = Sn
    S = Sn

    err = []

    count_es = 0
    min_err = np.inf
    norm_H = (H_true**2).sum() if H_true is not None else 0
    norm_S = (N*(N-1)) if S_true is not None else 0

    for i in range(max_iters):
        # Filter identification problem
        H = filter_id(Y, X, S, gamma, delta, Cy)
        H = H_prev if H is None else H

        # Graph identification
        S = graph_id(Sn, H, Cy, lambd, gamma, delta)
        S = S_prev if S is None else S

        # Check convergence   
        if H_true is not None and S_true is not None:
            # Early stopping is performed with variables error
            err_H = ((H - H_true)**2).sum() / norm_H
            err_S = ((S - S_true)**2).sum() / norm_S
            err.append(err_H + err_S)
            #print(f"estH_iter: {i=}, {err_H=}, {err_S=}, {err[i]=}")
        else:
            # Early stopping is performed with objective funtion
            ls_loss = ((Y - H@X)**2).sum()
            s_loss = np.abs(S-Sn).sum()
            commut_loss = ((S@H - H@S)**2).sum()
            commut_cy_loss = ((Cy@H - H@Cy)**2).sum()
            err.append(ls_loss + lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss)
        # print(f"Iter: {i} - err: {err[i]}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th and err[i] > err[i-1]:
            H_min = H
            S_min = S
            i_min = i
            #print(f'\t\tConvergence reached at iteration {i}')
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
            #print(f'\t\tES Convergence reached at iteration {i_min}')
            break

        gamma = inc_gamma*gamma if inc_gamma else gamma
        H_prev = H
        S_prev = S
    
    return i_min, H_min, S_min


def estH_iter_rew(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4, H_true=None, S_true=None):
    import warnings
    warnings.filterwarnings("ignore")
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
    norm_H = (H_true**2).sum() if H_true is not None else 0
    norm_S = (N*(N-1) / 2) if S_true is not None else 0

    for i in range(max_iters):
        # Filter identification problem
        H = filter_id(Y, X, S, gamma, delta, Cy)
        H = H_prev if H is None else H

        # Graph identification
        S = graph_id_rew(Sn, H, Cy, W1, W2, lambd, gamma, delta, beta)
        S = S_prev if S is None else S
        
        W1 = lambd / (np.abs(S - Sn) + delta1)
        W2 = beta / (S + delta2)

        if H_true is not None and S_true is not None:
            # Early stopping is performed with variables error
            err_H = ((H - H_true)**2).sum() / norm_H
            err_S = ((S - S_true)**2).sum() / norm_S
            err.append(err_H + err_S)
            #print(i, err_H, err_S, err[-1])
        else:
            # Early stopping is performed with objective funtion
            ls_loss = ((Y - H@X)**2).sum()
            s_loss = np.abs(S-Sn).sum()
            commut_loss = ((S@H - H@S)**2).sum()
            commut_cy_loss = ((Cy@H - H@Cy)**2).sum()
            err.append(ls_loss + lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss)
        # print(f"Iter: {i} - err: {err[i]}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th and err[i] > err[i-1]:
            H_min = H
            S_min = S
            i_min = i
            #print(f'Convergence reached at iteration {i}')
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
            #print(f"Convergence reached at iteration {i_min}", flush=True)
            break
        if inc_gamma:
            gamma = inc_gamma*gamma
        H_prev = H
        S_prev = S
    
    return i_min, H_min, S_min

def estH_denS(X, Y, Sn, Cy, params, verb=VERB):
    import warnings
    warnings.filterwarnings("ignore")
    N, M = X.shape

    gamma, delta = params

    # Denoise S
    S = cp.Variable((N,N), symmetric=True)

    s_loss = cp.sum(cp.abs(S - Sn))
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
        if verb:
            print("estH_denS -- Could not find optimal S -- Solver Error")
        try:
            prob.solve(verbose=False)
            if verb:
                print("Solver Error fixed")
        except:
            if verb:
                print("A second solver error")
            S = Sn

    if prob.status in ["optimal", "optimal_inaccurate"]:
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
    try:
        prob.solve()
    except cp.SolverError:
        if verb:
            print("estH_denS -- Could not find optimal H - SolverError")

    if prob.status in ["optimal", "optimal_inaccurate"]:
        H = H.value
    else:
        if verb:
            print("estH_denS -- Could not find optimal H")
        H = np.zeros((N,N))

    ls_loss = ((Y - H@X)**2).sum()
    s_loss = np.abs(S-Sn).sum()
    commut_loss = ((S@H - H@S)**2).sum()
    commut_cy_loss = ((Cy@H - H@Cy)**2).sum()
    err = ls_loss + s_loss + gamma*commut_loss + delta*commut_cy_loss

    return -1, H, S

def estH_unpertS(X, Y, S, Cy=None, params=None, H_true=None, S_true=None):
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

def estH_ls(X, Y, S, Cy=None, params=None, H_true=None, S_true=None):
    """
    Estimation of the H matrix via least squares
    """

    H = Y @ np.linalg.pinv(X)

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


def estH_analyticalSol(X, Y, Sn, Cy, params, max_iters=20, th=1e-3):
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


# Baselines
def estH_tls_sem(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4, H_true=None, S_true=None, verb=VERB):
    N = Sn.shape[0]
    lambd1, lambd2 = params

    count_es = 0
    min_err = np.inf
    norm_H = (H_true**2).sum() if H_true is not None else 0
    norm_S = (N*(N-1) / 2) if S_true is not None else 0

    Y_aux = Y.copy()

    err = []

    for i in range(max_iters):
        Delta = cp.Variable((N,N), symmetric=True)

        ls_loss = cp.sum_squares(Y_aux - (Sn-Delta)@Y_aux - X)

        sparsity_loss = cp.sum(cp.abs(Delta))

        obj = ls_loss + lambd1*sparsity_loss
        const = [cp.diag(Delta) == 0]

        prob = cp.Problem(cp.Minimize(obj), const)
        try:
            prob.solve()
        except cp.SolverError:
            if verb:
                print("estH_tls_sem -- Could not find optimal Delta - SolverError")
        
        if prob.status in ["optimal", "optimal_inaccurate"]:
            S = Sn - Delta.value
        else:
            S = Sn
        S_tilde = np.eye(N) - S
        H = np.linalg.inv(S_tilde)

        # Analytical solution
        Y_aux, _, _, _ = np.linalg.lstsq(S_tilde @ S_tilde + lambd2*np.eye(N), S_tilde @ X + lambd2*Y, rcond=None)

        if H_true is not None and S_true is not None:
            # Early stopping is performed with variables error
            err_H = ((H - H_true)**2).sum() / norm_H
            err_S = ((S - S_true)**2).sum() / norm_S
            err.append(err_H + err_S)
            #print(i, err_H, err_S, err[-1])
        else:
            # Early stopping is performed with objective funtion
            ls_loss = ((Y - S@Y - X)**2).sum()
            s_loss = np.abs(S).sum()
            y_loss = ((Y-Y_aux)**2).sum()
            err.append(ls_loss + lambd1*s_loss + lambd2*y_loss)
        # print(f"Iter: {i} - err: {err[i]}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th and err[i] > err[i-1]:
            H_min = H
            S_min = S
            i_min = i
            #print(f'Convergence reached at iteration {i}')
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
            #print(f"Convergence reached at iteration {i_min}", flush=True)
            break

    return i_min, H_min, S_min


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
    diff_S = np.zeros(max_iters-1)
    diff_H = np.zeros(max_iters-1)
    for i in range(max_iters):
        # Filter identification step
        Z = np.kron(S@S, np.eye(N)) + np.kron(np.eye(N), S@S) - 2*np.kron(S, S)
        H = (np.linalg.inv(X_kron + gamma*Z)@Y_kron).reshape((N,N), order='F')

        # Graph identification
        S = graph_id(Sn, H, Cy, lambd, gamma, delta)
        if S is None:
            print(f"rfi_debug - Iter {i} - S is None")
            S = S_prev

        if inc_gamma:
            gamma = inc_gamma*gamma

        err_H[i] = (np.linalg.norm(H - H_true, 'fro')/norm_H)**2
        err_S[i] = (np.linalg.norm(S - S_true, 'fro')/norm_S)**2

        if i > 0:
            diff_H[i-1] = ((H-H_prev)**2).sum()
            diff_S[i-1] = ((S-S_prev)**2).sum()
        H_prev = H
        S_prev = S

    return H, S, err_H, err_S, diff_H, diff_S

##########   DEBUG METHODS   ########## 
def rfi_cvx_debug(X, Y, Sn, Cy, params, H_true, S_true, max_iters=20):
    # Initialization
    lambd, gamma, delta, inc_gamma = params
    N, M = X.shape
    S_prev = S = Sn
    H_prev = Sn

    # Precomputing quantities
    norm_H = np.linalg.norm(H_true,'fro')
    norm_S = np.linalg.norm(S_true, 'fro')

    err_S = np.zeros(max_iters)
    err_H = np.zeros(max_iters)
    diff_S = np.zeros(max_iters-1)
    diff_H = np.zeros(max_iters-1)
    for i in range(max_iters):
        # Filter identification problem
        H = filter_id(Y, X, S, gamma, delta, Cy)
        H = H_prev if H is None else H

        # Graph identification
        S = graph_id(Sn, H, Cy, lambd, gamma, delta)
        if S is None:
            print(f"rfi_cvx_debug - Iter {i} - S is None")
            S = S_prev

        if inc_gamma:
            gamma = inc_gamma*gamma

        err_H[i] = (np.linalg.norm(H - H_true, 'fro')/norm_H)**2
        err_S[i] = (np.linalg.norm(S - S_true, 'fro')/norm_S)**2

        if i > 0:
            diff_H[i-1] = ((H-H_prev)**2).sum()
            diff_S[i-1] = ((S-S_prev)**2).sum()
        H_prev = H
        S_prev = S

    return H, S, err_H, err_S, diff_H, diff_S

##########   DEBUG METHODS   ########## 
def rfi_rew_debug(X, Y, Sn, Cy, params, H_true, S_true, max_iters=20):
    # Initialization
    lambd, gamma, delta, beta, inc_gamma = params
    N, M = X.shape
    S_prev = S = Sn
    H_prev = Sn

    # Precomputing quantities
    norm_H = np.linalg.norm(H_true,'fro')
    norm_S = np.linalg.norm(S_true, 'fro')

    W1 = np.ones((N,N))
    W2 = np.ones((N,N))
    delta1 = 1e-3
    delta2 = 1e-3

    norm_H = (H_true**2).sum() if H_true is not None else 0
    norm_S = np.linalg.norm(S_true, 'fro') if S_true is not None else 0

    err_S = np.zeros(max_iters)
    err_H = np.zeros(max_iters)
    diff_S = np.zeros(max_iters-1)
    diff_H = np.zeros(max_iters-1)
    for i in range(max_iters):
        # Filter identification problem
        H = filter_id(Y, X, S, gamma, delta, Cy)
        H = H_prev if H is None else H

        # Graph identification
        S = graph_id_rew(Sn, H, Cy, W1, W2, lambd, gamma, delta, beta)
        if S is None:
            print(f"rfi_rew_debug - Iter {i} - S is None")
            S = S_prev

        W1 = lambd / (np.abs(S - Sn) + delta1)
        W2 = beta / (S + delta2)

        if inc_gamma:
            gamma = inc_gamma*gamma

        err_H[i] = (np.linalg.norm(H - H_true, 'fro')/norm_H)**2
        err_S[i] = (np.linalg.norm(S - S_true, 'fro')/norm_S)**2

        if i > 0:
            diff_H[i-1] = ((H-H_prev)**2).sum()
            diff_S[i-1] = ((S-S_prev)**2).sum()
        H_prev = H
        S_prev = S

    return H, S, err_H, err_S, diff_H, diff_S