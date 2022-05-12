import cvxpy as cp
import numpy as np
from scipy.linalg import khatri_rao

def estHs_iter(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4):
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

    for i in range(max_iters):
        # Filter identification problem

        # Creating Hs as a list as CVXpy does not support N-dimensional arrays
        Hs = [cp.Variable((N, N), symmetric=True) for _ in range(T)]

        ls_loss = cp.sum([cp.sum_squares(Y[i,:,:] - Hs[i]@X[i,:,:]) for i in range(T)])
        commut_loss = cp.sum([cp.sum_squares(Hs[i]@S - S@Hs[i]) for i in range(T)])
        commut_cy_loss = cp.sum([cp.sum_squares(Hs[i]@Cy[i,:,:] - Cy[i,:,:]@Hs[i]) for i in range(T)])

        obj = ls_loss + gamma*commut_loss + delta*commut_cy_loss

        prob = cp.Problem(cp.Minimize(obj))
        try:
            prob.solve()
        except cp.SolverError:
            print("estHs_iter -- Could not find optimal H -- Solver Error")
            Hs = Hs_prev

        if prob.status == "optimal":
            Hs = np.array([Hs[i].value for i in range(T)])
            Hs_prev = Hs.copy()
        else:
            print("estHs_iter -- Could not find optimal H")
            Hs = Hs_prev

        # Havg = np.mean(Hs, 0)

        # Graph identification
        S = cp.Variable((N,N), symmetric=True)

        s_loss = cp.norm(S - Sn, 1)
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
            #print("estHs_iter -- Could not find optimal S -- Solver Error")
            S = S_prev

        if prob.status == "optimal":
            S = S.value
        else:
            #print("estHs_iter -- Could not find optimal S")
            S = S_prev

        ls_loss = ((Y - Hs@X)**2).sum()
        s_loss = np.abs(S-Sn).sum()
        commut_loss = ((S@Hs - Hs@S)**2).sum()
        commut_cy_loss = ((Cy@Hs - Hs@Cy)**2).sum()
        err.append(ls_loss + lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss)
        # print(f"Iter: {i} - err: {err[i]}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th:
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
    
    return i_min, Hs_min, S_min

def estHs_iter_rew(X, Y, Sn, Cy, params, max_iters=20, th=1e-3, patience=4):

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

    for i in range(max_iters):
        # Filter identification problem

        # Creating Hs as a list as CVXpy does not support N-dimensional arrays
        Hs = [cp.Variable((N, N), symmetric=True) for _ in range(T)]

        ls_loss = cp.sum([cp.sum_squares(Y[i,:,:] - Hs[i]@X[i,:,:]) for i in range(T)])
        commut_loss = cp.sum([cp.sum_squares(Hs[i]@S - S@Hs[i]) for i in range(T)])
        commut_cy_loss = cp.sum([cp.sum_squares(Hs[i]@Cy[i,:,:] - Cy[i,:,:]@Hs[i]) for i in range(T)])

        obj = ls_loss + gamma*commut_loss + delta*commut_cy_loss

        prob = cp.Problem(cp.Minimize(obj))
        try:
            prob.solve()
        except cp.SolverError:
            print("estHs_iter_rew -- Could not find optimal H -- Solver Error")
            Hs = Hs_prev

        if prob.status == "optimal":
            Hs = np.array([Hs[i].value for i in range(T)])
            Hs_prev = Hs.copy()
        else:
            print("estHs_iter_rew -- Could not find optimal H")
            Hs = Hs_prev

        #Havg = np.mean(Hs, 0)

        # Graph identification
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
            S = S_prev
            #print("estHs_iter_rew -- Could not find optimal S - Solver Error")
        except cp.DCPError:
            print("estHs_iter_rew -- Could not find optimal S - DCPError")
            raise RuntimeError()

        if prob.status == "optimal":
            S = S.value
        else:
            #print("estHs_iter_rew -- Could not find optimal S")
            S = S_prev
        
        W1 = lambd / (np.abs(S - Sn) + delta1)
        W2 = beta / (S + delta2)

        ls_loss = ((Y - Hs@X)**2).sum()
        s_loss = np.abs(S-Sn).sum()
        commut_loss = ((S@Hs - Hs@S)**2).sum()
        commut_cy_loss = ((Cy@Hs - Hs@Cy)**2).sum()
        err.append(ls_loss + lambd*s_loss + gamma*commut_loss + delta*commut_cy_loss)
        # print(f"Iter: {i} - err: {err[i]}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th:
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
    
    return i_min, Hs_min, S_min

def estHs_denS(X, Y, Sn, Cy, params):

    import warnings
    warnings.filterwarnings("ignore")
    T, N, M = X.shape

    gamma, delta = params

    # Denoise S
    S = cp.Variable((N,N), symmetric=True)

    s_loss = cp.norm(S - Sn, 1)
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

    if prob.status == "optimal":
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

    if prob.status == "optimal":
        Hs = np.array([Hs[i].value for i in range(T)])
    else:
        print("estHs_denS -- Could not find optimal H")
        Hs = np.zeros((T, N, N))

    ls_loss = ((Y - Hs@X)**2).sum()
    s_loss = np.abs(S-Sn).sum()
    commut_loss = ((S@Hs - Hs@S)**2).sum()
    commut_cy_loss = ((Cy@Hs - Hs@Cy)**2).sum()
    err = ls_loss + s_loss + gamma*commut_loss + delta*commut_cy_loss

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