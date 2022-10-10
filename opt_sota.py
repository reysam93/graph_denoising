
import numpy as np
import cvxpy as cp
from data import obtain_filter_coefs

VERB = False

def filter_id_lls(X, Y, S, K, verb=VERB):

    Spow = np.array([np.linalg.matrix_power(S, k) for k in range(K)])

    h = cp.Variable(K)

    H = cp.sum([h[k]*Spow[k,:,:] for k in range(K)])
    obj = cp.sum_squares(Y - H @ X)

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
        return h.value
    
    if verb:
        print(f"WARNING: problem status: {prob.status}")
    return None

def graph_score_func(X, Y, S, h, Spow=None):
    K = h.shape[0]
    if Spow is None:
        Spow = np.array([np.linalg.matrix_power(S, k) for k in range(K)])
    
    H = np.array([h[k]*Spow[k,:,:] for k in range(K)])
    return np.sum((Y - H @ X)**2)

def calc_grad(X, Y, S, h):
    K = h.shape[0]
    Spow = np.array([np.linalg.matrix_power(S, k) for k in range(2*K-1)])
    term_1 = np.zeros(S.shape)
    for k in range(1, K):
        int_term = [Spow[r,:,:] @ X @ Y.T @ Spow[k-r-1,:,:] for r in range(k-1)]
        term_1 += h[k] * np.sum(np.array(int_term), 0).T

    term_2 = np.zeros(S.shape)
    for k1 in range(K):
        for k2 in range(K):
            if k1 == k2 == 0:
                continue
            int_term = [Spow[r,:,:] @ X @ X.T @ Spow[k1+k2-r-1,:,:] for r in range(k1+k2-1)]
            term_2 += h[k1] * h[k2] * np.sum(np.array(int_term), 0).T
    
    return -2 * term_1 + term_2

def graph_id_step(X, Y, S_l, h, rho, supp_A, Spow=None, verb=VERB):
    f_S = graph_score_func(X, Y, S_l, h)

    grad_f = calc_grad(X, Y, S_l, h)

    S = cp.Variable(S_l.shape, symmetric=True)
    f_hat_S = f_S + cp.trace(grad_f.T @ (S - S_l))

    trust_reg = [
        S >= 0,
        cp.diag(S) == 0,
        cp.abs(S - S_l) <= rho
    ]
    if supp_A is not None:
        trust_reg.append(S[supp_A] == 0)

    prob = cp.Problem(cp.Minimize(f_hat_S), trust_reg)

    try:
        prob.solve()
    except cp.SolverError:
        if verb:
            print("WARNING: Could not find optimal S -- Solver Error")
        
    if prob.status in ["optimal", "optimal_inaccurate"]:
        return S.value
    
    if verb:
        print(f"WARNING: problem status: {prob.status}")
    return None

def find_step(X, Y, S_hat, S_l, h, verb=VERB):

    delta_l = S_hat - S_l

    alpha_vals = [1e-4, 1e-3, 1e-2, 1]

    f_S = np.array([graph_score_func(X, Y, S_hat + a*delta_l, h) for a in alpha_vals])
        
    return alpha_vals[np.argmin(f_S)]

def graph_id_scp(X, Y, Sn, h, rho, supp_A, max_iters=1000, th=1e-2, patience=10, verb=VERB):

    S = Sn.copy()
    f_S = np.zeros(max_iters)

    count_es = 0
    min_err = np.inf

    S_prev = Sn.copy()

    for i in range(max_iters):
        S_hat = graph_id_step(X, Y, S, h, rho, supp_A)
        S_hat = S_prev if S_hat is None else S_hat

        alpha = find_step(X, Y, S_hat, S, h)

        S += alpha*(S_hat - S)

        f_S[i] = graph_score_func(X, Y, S, h)

        if i > 0 and np.abs(f_S[i] - f_S[i-1]) < th and f_S[i] > f_S[i-1]:
            h_min = h
            S_min = S
            i_min = i
            #print(f'\t\tConvergence reached at iteration {i}')
            break
        if f_S[i] > min_err:
            count_es += 1
        else:
            min_err = f_S[i]
            h_min = h.copy()
            S_min = S.copy()
            i_min = i
            count_es = 0
        
        if count_es == patience:
            #print(f'\t\tES Convergence reached at iteration {i_min}')
            break

        S_prev = S_hat

    return S_min
        


def estH_llsscp(X, Y, Sn, Cy, params, K=5, max_iters=20, th=1e-3, patience=4, H_true=None, S_true=None, use_support=False):
    import warnings
    warnings.filterwarnings("ignore")
    rho = params

    N, M = X.shape
    S_prev = Sn
    h_prev = np.ones(K)
    S = Sn

    norm_S = (N*(N-1)) if S_true is not None else 0
    if H_true is not None:
        h_true = obtain_filter_coefs(S_true, H_true, K)
        norm_h = (h_true**2).sum()
        supp_A = np.where(S_true == 0)
    else:
        supp_A = np.where(Sn == 0)

    if not use_support:
        supp_A = None

    err = []

    count_es = 0
    min_err = np.inf

    for i in range(max_iters):
        # Filter identification problem
        h = filter_id_lls(X, Y, S, K)
        h = h_prev if h is None else h

        # Graph identification
        S = graph_id_scp(X, Y, Sn, h, rho, supp_A)
        S = S_prev if S is None else S

        # Check convergence        
        if H_true is not None and S_true is not None:
            # Early stopping is performed with variables error
            err_h = ((h - h_true)**2).sum() / norm_h
            err_S = ((S - S_true)**2).sum() / norm_S
            err.append(err_h + err_S)
            #print(f"estH_iter: {i=}, {err_H=}, {err_S=}, {err[i]=}")
        else:
            err.append(0.)
        # print(f"Iter: {i} - err: {err[i]}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th and err[i] > err[i-1]:
            h_min = h
            S_min = S
            i_min = i
            #print(f'\t\tConvergence reached at iteration {i}')
            break
        if err[i] > min_err:
            count_es += 1
        else:
            min_err = err[i]
            h_min = h.copy()
            S_min = S.copy()
            i_min = i
            count_es = 0
        
        if count_es == patience:
            #print(f'\t\tES Convergence reached at iteration {i_min}')
            break

        h_prev = h
        S_prev = S
    

    #thres = (np.max(S_min) + np.min(S_min))/2
    #S_est = np.where(S_min > thres, 1., 0.)
    H_min = np.sum(np.array([h_min[k]*np.linalg.matrix_power(S_min, k) for k in range(K)]), 0)
    return i_min, H_min, S_min
