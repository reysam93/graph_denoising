import cvxpy as cp
import numpy as np
from scipy.linalg import khatri_rao


def estH_iter(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4):
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

    for i in range(max_iters):
        # Filter identification problem
        H = cp.Variable((N,N), symmetric=True)

        ls_loss = cp.sum_squares(Y - H@X)
        commut_loss = cp.sum_squares(H@S - S@H)
        commut_cy_loss = cp.sum_squares(H@Cy - Cy@H)

        obj = ls_loss + gamma*commut_loss + delta*commut_cy_loss

        prob = cp.Problem(cp.Minimize(obj))
        try:
            prob.solve()
        except cp.SolverError:
            print("estH_iter -- Could not find optimal H -- Solver Error")
            H = H_prev

        if prob.status == "optimal":
            H = H.value
        else:
            print("estH_iter -- Could not find optimal H")
            H = H_prev

        # Graph identification
        S = cp.Variable((N,N), symmetric=True)

        s_loss = cp.norm(S - Sn, 1)
        commut_loss = cp.sum_squares(H@S - S@H)
        commut_cy_loss = cp.sum_squares(S@Cy - Cy@S)

        obj = lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss

        constraints = [
            S >= 0,
            cp.diag(S) == 0
        ]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        try:
            prob.solve()
        except cp.SolverError:
            #print("estH_iter -- Could not find optimal S -- Solver Error")
            S = S_prev

        if prob.status == "optimal":
            S = S.value
        else:
            #print("estH_iter -- Could not find optimal S")
            S = S_prev

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

def estH_iter_rew(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4):

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

    for i in range(max_iters):
        # Filter identification problem
        H = cp.Variable((N,N), symmetric=True)

        ls_loss = cp.sum_squares(Y - H@X)
        commut_loss = cp.sum_squares(H@S - S@H)
        commut_cy_loss = cp.sum_squares(H@Cy - Cy@H)

        obj = ls_loss + gamma*commut_loss + delta*commut_cy_loss

        prob = cp.Problem(cp.Minimize(obj))
        try:
            prob.solve()
        except cp.SolverError:
            print("estH_iter_rew -- Could not find optimal H -- SolverError")
            H = H_prev

        if prob.status == "optimal":
            H = H.value
        else:
            print("estH_iter_rew -- Could not find optimal H")
            H = H_prev

        # Graph identification
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
            #print("estH_iter_rew -- Could not find optimal S -- Solver Error")
            S = S_prev
        except cp.DCPError:
            print("estH_iter_rew -- Could not find optimal S -- DCP Error")
            raise RuntimeError()

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

    import warnings
    warnings.filterwarnings("ignore")
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

def estH_unpertS(X, Y, S, Cy, params):
    import warnings
    warnings.filterwarnings("ignore")
    _, eigvecs = np.linalg.eigh(S)

    Z = khatri_rao(X.T @ eigvecs, eigvecs)
    h_freq, _, _, _ = np.linalg.lstsq(Z, Y.flatten('F'))
    H = eigvecs @ np.diag(h_freq) @ eigvecs.T
    return -1, H, S