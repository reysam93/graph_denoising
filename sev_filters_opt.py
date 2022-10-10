import cvxpy as cp
import numpy as np
from opt import filter_id

from scipy.linalg import khatri_rao

VERB = False

def filter_id_sevH(Y, X, S, gamma, delta, Cy, verb=VERB):
    """
    Performs the filter identification step of the robust filter identification algorithm.
    It estimates H using cvx, not the analytical solution.
    """

    T, N, M = X.shape
    Hs = [cp.Variable((N, N), symmetric=True) for _ in range(T)]

    ls_loss = cp.sum([cp.sum_squares(Y[i,:,:] - Hs[i]@X[i,:,:]) for i in range(T)])
    commut_loss = cp.sum([cp.sum_squares(Hs[i]@S - S@Hs[i]) for i in range(T)])
    commut_cy_loss = cp.sum([cp.sum_squares(Hs[i]@Cy[i,:,:] - Cy[i,:,:]@Hs[i]) for i in range(T)])

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
        return np.array([Hs[i].value for i in range(T)])
    
    if verb:
        print(f"WARNING: problem status: {prob.status}")
    return None

def graph_id(Sn, Hs, Cy, lambd, gamma, delta, verb=VERB):
    """
    Performs the filter identification step of the robust filter identification algorithm
    """
    T, N, _ = Hs.shape
    S = cp.Variable((N,N), symmetric=True)

    s_loss = cp.sum(cp.abs(S - Sn))
    commut_loss = cp.sum([cp.sum_squares(Hs[i,:,:]@S - S@Hs[i,:,:]) for i in range(T)])
    #commut_loss = cp.sum_squares(Havg@S - S@Havg)
    #commut_cy_loss = cp.sum_squares(S@Cy - Cy@S)
    commut_cy_loss = cp.sum([cp.sum_squares(Cy[i,:,:]@S - S@Cy[i,:,:]) for i in range(T)])

    obj = lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss

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

def graph_id_rew(Sn, Hs, Cy, W1, W2, lambd, gamma, delta, beta, verb=VERB):
    """
    Performs the filter identification step of the robust filter identification algorithm
    with the reweighted alternative
    """
    T, N, _ = Hs.shape
    S = cp.Variable((N,N), symmetric=True)

    sn_loss = cp.sum(cp.multiply(W1, cp.abs(S - Sn)))
    s_loss = cp.sum(cp.multiply(W2, cp.abs(S)))
    commut_loss = cp.sum([cp.sum_squares(Hs[i,:,:]@S - S@Hs[i,:,:]) for i in range(T)])
    #commut_loss = cp.sum_squares(Havg@S - S@Havg)
    #commut_cy_loss = cp.sum_squares(S@Cy - Cy@S)
    commut_cy_loss = cp.sum([cp.sum_squares(Cy[i,:,:]@S - S@Cy[i,:,:]) for i in range(T)])

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


def estHs_iter(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4, Hs_true=None, S_true=None):
    import warnings
    warnings.filterwarnings("ignore")

    lambd, gamma, delta, inc_gamma = params

    T, N, M = X.shape
    S_prev = Sn
    Hs_prev = np.array([Sn for _ in range(T)])
    S = Sn

    err = []

    count_es = 0
    min_err = np.inf

    norm_Hs = (Hs_true**2).sum((1,2)) if Hs_true is not None else 0
    norm_S = (N*(N-1)) if S_true is not None else 0

    Hs = np.zeros((T,N,N))

    for i in range(max_iters):
        # Filter identification problem

        for t in range(T):
            H_id = filter_id(Y[t,:,:], X[t,:,:], S, gamma, delta, Cy[t,:,:])
            Hs[t,:,:] = Hs_prev[t,:,:] if H_id is None else H_id

        # Havg = np.mean(Hs, 0)

        # Graph identification
        S = graph_id(Sn, Hs, Cy, lambd, gamma, delta)
        S = S_prev if S is None else S

        if Hs_true is not None and S_true is not None:
            # Early stopping is performed with variables error
            err_Hs = np.median(((Hs - Hs_true)**2).sum((1,2)) / norm_Hs)
            err_S = ((S - S_true)**2).sum() / norm_S
            err.append(err_Hs + err_S)
            #print(f"Sev: {i=}, {err_Hs=}, {err_S=}, {err[i]=}")
        else:
            ls_loss = ((Y - Hs@X)**2).sum()
            s_loss = np.abs(S-Sn).sum()
            commut_loss = ((S@Hs - Hs@S)**2).sum()
            commut_cy_loss = ((Cy@Hs - Hs@Cy)**2).sum()
            err.append(ls_loss + lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss)
        # print(f"Iter: {i} - err: {err[i]}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th and err[i] > err[i-1]:
            Hs_min = Hs
            S_min = S
            i_min = i
            break
        if err[i] > min_err:
            count_es += 1
        else:
            min_err = err[i]
            Hs_min = Hs.copy()
            S_min = S.copy()
            i_min = i
            count_es = 0
        
        if count_es == patience:
            break
        gamma = inc_gamma*gamma if inc_gamma else gamma
        Hs_prev = Hs
        S_prev = S
    
    return i_min, Hs_min, S_min

def estHs_iter_rew(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4, Hs_true=None, S_true=None):

    import warnings
    warnings.filterwarnings("ignore")

    lambd, gamma, delta, beta, inc_gamma = params

    T, N, M = X.shape
    S_prev = Sn
    Hs_prev = np.array([Sn for _ in range(T)])
    S = Sn

    err = []

    W1 = np.ones((N,N))
    W2 = np.ones((N,N))
    delta1 = 1e-3
    delta2 = 1e-3

    count_es = 0
    min_err = np.inf

    norm_Hs = (Hs_true**2).sum((1,2)) if Hs_true is not None else 0
    norm_S = (N*(N-1)) if S_true is not None else 0

    Hs = np.zeros((T,N,N))

    for i in range(max_iters):
        # Filter identification problem

        for t in range(T):
            H_id = filter_id(Y[t,:,:], X[t,:,:], S, gamma, delta, Cy[t,:,:])
            Hs[t,:,:] = Hs_prev[t,:,:] if H_id is None else H_id

        # Graph identification
        S = graph_id_rew(Sn, Hs, Cy, W1, W2, lambd, gamma, delta, beta)
        S = S_prev if S is None else S
        
        W1 = lambd / (np.abs(S - Sn) + delta1)
        W2 = beta / (S + delta2)

        if Hs_true is not None and S_true is not None:
            # Early stopping is performed with variables error
            err_Hs = (((Hs - Hs_true)**2).sum((1,2)) / norm_Hs).mean()
            err_S = ((S - S_true)**2).sum() / norm_S
            err.append(err_Hs + err_S)
            #print(i, err_H, err_S, err[i])
        else:
            ls_loss = ((Y - Hs@X)**2).sum()
            s_loss = np.abs(S-Sn).sum()
            commut_loss = ((S@Hs - Hs@S)**2).sum()
            commut_cy_loss = ((Cy@Hs - Hs@Cy)**2).sum()
            err.append(ls_loss + lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss)
        # print(f"Iter: {i} - err: {err[i]}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th and err[i] > err[i-1]:
            Hs_min = Hs
            S_min = S
            i_min = i
            break
        if err[i] > min_err:
            count_es += 1
        else:
            min_err = err[i]
            Hs_min = Hs.copy()
            S_min = S.copy()
            i_min = i
            count_es = 0
        
        if count_es == patience:
            break
        if inc_gamma:
            gamma = inc_gamma*gamma

        Hs_prev = Hs
        S_prev = S
    
    return i_min, Hs_min, S_min

def estHs_denS(X, Y, Sn, Cy, params, verb=VERB):

    import warnings
    warnings.filterwarnings("ignore")
    T, N, M = X.shape

    gamma, delta = params

    # Denoise S
    S = cp.Variable((N,N), symmetric=True)

    s_loss = cp.sum(cp.abs(S - Sn))
    #commut_cy_loss = cp.sum_squares(S@Cy - Cy@S)
    commut_cy_loss = cp.sum([cp.sum_squares(Cy[i,:,:]@S - S@Cy[i,:,:]) for i in range(T)])

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
        #print("estHs_denS -- Could not find denoised version of S - SolverError")
        S = Sn

    if prob.status in ["optimal", "optimal_inaccurate"]:
        S = S.value
    else:
        #print("estHs_denS -- Could not find denoised version of S - Not optimal")
        S = Sn

    # Compute H from S
    Hs = [cp.Variable((N,N), symmetric=True) for _ in range(T)]

    ls_loss = cp.sum([cp.sum_squares(Y[i,:,:] - Hs[i]@X[i,:,:]) for i in range(T)])
    try:
        commut_loss = cp.sum([cp.sum_squares(Hs[i]@S - S@Hs[i]) for i in range(T)])
        commut_cy_loss = cp.sum([cp.sum_squares(Hs[i]@Cy[i,:,:] - Cy[i,:,:]@Hs[i]) for i in range(T)])
    except ValueError:
        raise RuntimeError("Value Error when defining Commut Loss")

    obj = ls_loss + gamma*commut_loss + delta*commut_cy_loss

    #const = [commut_loss <= 0]

    prob = cp.Problem(cp.Minimize(obj))#, const)
    try:
        prob.solve()
    except cp.SolverError:
        print("estHs_denS -- Could not find optimal H -- Solver Error")
        Hs = np.zeros((T, N, N))

    if prob.status in ["optimal", "optimal_inaccurate"]:
        Hs = np.array([Hs[i].value for i in range(T)])
    else:
        if verb:
            print("estHs_denS -- Could not find optimal H")
        Hs = np.zeros((T, N, N))

    #ls_loss = ((Y - Hs@X)**2).sum()
    #s_loss = np.abs(S-Sn).sum()
    #commut_loss = ((S@Hs - Hs@S)**2).sum()
    #commut_cy_loss = ((Cy@Hs - Hs@Cy)**2).sum()
    #err = ls_loss + s_loss + gamma*commut_loss + delta*commut_cy_loss

    return -1, Hs, S

def estHs_unpertS(X, Y, S, Cy, params):
    import warnings
    warnings.filterwarnings("ignore")
    _, eigvecs = np.linalg.eigh(S)
    Hs = []

    for i in range(X.shape[0]):
        Z = khatri_rao(X[i,:,:].T @ eigvecs, eigvecs)
        h_freq, _, _, _ = np.linalg.lstsq(Z, Y[i,:,:].flatten('F'))
        Hs.append(eigvecs @ np.diag(h_freq) @ eigvecs.T)

    return -1, np.array(Hs), S