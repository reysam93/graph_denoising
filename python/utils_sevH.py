import sys
sys.path.append('..')

import data
import opt
import sev_filters_opt

import numpy as np

from joblib import Parallel, delayed, cpu_count

def obtain_filter_coefs_sevH(S, Hs, K):
    """
    Obtain the coefficients h used to generate several graph filters Hs, using the GSO
    given as an argument
    """
    T = Hs.shape[0]
    Svals, Svecs = np.linalg.eigh(S)
    Psi = np.fliplr(np.vander(Svals))
    Psi_inv = np.linalg.pinv(Psi[:,:K])

    h_bar_mat = Svecs.T @ Hs @ Svecs # Should be T x N x N
    h_bar = np.array([np.diag(h_bar_mat[t,:,:]) for t in range(T)]) # should be T x N
    h = Psi_inv @ h_bar.T # should be K x T
    assert h.shape == (K, T)

    return h.T

def test(X, Y, Sn, S, Hs, h, K, Cy, Cy_samp, exps, args, S_for_coefs="binarized"):
    err_H = np.zeros((len(exps)))
    err_S = np.zeros((len(exps)))
    err_H_coefs = np.zeros((len(exps)))
    T, N, _ = Hs.shape

    #norm_h = np.sqrt((Hs**2).sum((1,2)))
    #norm_A = np.sqrt((S**2).sum())
    norm_h = (Hs**2).sum((1,2))
    norm_A = (S**2).sum()
    h_true = obtain_filter_coefs_sevH(S, Hs, K)
    #assert np.allclose(h, h_true) # Care with normalization
    import warnings
    warnings.filterwarnings('ignore')

    for i, exp in enumerate(exps):

        if exp["cy"] == "real" or exp["cy"] == "non-st":
            Cy_exp = Cy
        elif exp["cy"] == "samp":
            Cy_exp = Cy_samp
        else:
            raise NotImplementedError("Choose either real or samp covariance")

        params = args[exp["cy"]].get(exp["func"], [])

        # if "iter" in exp["func"] or "tls" in exp['func']: # Not being used right now, arguments are given directly. Will not work with denS for example
        #     kwargs = {"S_true": S}
        #     if exp['joint']:
        #         kwargs["H_true"] = Hs
        # else:
        #     print("WARNING - kwargs not adapted to several H")
        #     kwargs = {}
        
        if exp['joint']:
            iter, H_est, S_est = getattr(sev_filters_opt, exp["func"])(X, Y, Sn, Cy_exp, params, Hs_true=Hs, S_true=S)
        else:
            H_est = np.zeros((T, N, N))
            S_est = np.zeros((T, N, N))
            for t in range(T):
                iter, H_est[t,:,:], S_est[t,:,:] = getattr(opt, exp["func"])(X[t,:,:], Y[t,:,:], Sn, Cy_exp[t,:,:], params, H_true=Hs[t,:,:], S_true=S)
            S_est = np.mean(S_est, 0)

        #print(exp['func'], ((Hs - H_est)**2).sum((1,2)) / norm_h, flush=True)
        err_H[i] = np.mean(((Hs - H_est)**2).sum((1,2)) / norm_h)
        err_S[i] = ((S - S_est)**2).sum()/norm_A
        #err_S[i] = (((S / norm_A) - S_est/(S_est**2).sum())**2).sum()

        if S_for_coefs == "binarized":
            # Binarization
            thres = (np.max(S_est) + np.min(S_est))/2
            S_est_coefs = np.where(S_est > thres, 1., 0.)
        elif S_for_coefs == "scaled":
            # Scale so maximum value is 1
            S_est_coefs = S_est / np.max(np.abs(S_est))
        elif S_for_coefs == "norm":
            # Scale so both S and S_est have the same norm
            S_est_coefs = (S_est / (S_est**2).sum()) * norm_A
            #assert np.allclose((S_est_coefs**2).sum(), norm_A)

        #err_S[i] = ((S - S_est_coefs)**2).sum()/norm_A

        h_est = obtain_filter_coefs_sevH(S_est_coefs, H_est, K)
        h_est = h_est / (h_est**2).sum()
        err_H_coefs[i] = np.median((h - h_est)**2)
        #print(f"Function {f} took {iter} iterations on covariance {cy_use}")
        
    return err_H, err_S, err_H_coefs


def objective_sevH(p_n, M, K, eps, T, exps, args, n_graphs, N, g_params, neg_coefs=False, exp_coefs=True, sort_h=False, norm_S=False, norm_H=False, pert_type="rewire", n_procs=cpu_count()):

    err_H = np.zeros((n_graphs, len(exps)))
    err_S = np.zeros((n_graphs, len(exps)))
    err_H_coefs = np.zeros((n_graphs, len(exps)))

    with Parallel(n_jobs = n_procs) as parallel: #, backend='multiprocessing'

        pert = pert_type if pert_type == "rewire" else "creat-dest"

        if pert_type == "creat":
            creat = 2*eps
            dest = 0.
        elif pert_type == "dest":
            dest = 2*eps
            creat = 0.
        else:
            creat = None
            dest = None

        funcs = []
        for i in range(n_graphs):
            X, Y, Cy, Cy_samp, Hs, S, Sn, h = data.gen_data_sev_H(N, M, T, g_params, p_n, eps, K, neg_coefs=neg_coefs, exp_coefs=exp_coefs, sort_h=sort_h, norm_S=norm_S, norm_H=norm_H, pert_type=pert, creat=creat, dest=dest)
                
            funcs.append(delayed(test)(X, Y, Sn, S, Hs, h, K, Cy, Cy_samp, exps, args))
        
        results = parallel(funcs)

        for i in range(n_graphs):
            err_H[i,:], err_S[i,:], err_H_coefs[i,:] = results[i]

    return {
        'med_H': np.median(err_H, 0),
        'std_H': np.std(err_H, 0),
        'mean_H': np.mean(err_H, 0),
        'mean_S': np.mean(err_S, 0),
        'med_S': np.median(err_S, 0),
        'std_S': np.std(err_S, 0),
        'med_H_coefs': np.median(err_H_coefs, 0),
        'std_H_coefs': np.std(err_H_coefs, 0),
        'mean_H_coefs': np.mean(err_H_coefs, 0)
    }


def objective_sem(p_n, M, K, eps, exps, args, n_graphs, N, g_params, neg_coefs=False, exp_coefs=True, sort_h=False, norm_S=False, norm_H=False, pert_type="rewire", n_procs=cpu_count()):
    err_H = np.zeros((n_graphs, 2, len(exps)))
    err_S = np.zeros((n_graphs, 2, len(exps)))
    err_H_coefs = np.zeros((n_graphs, 2, len(exps)))
    err_h_bar = np.zeros((n_graphs, 2, len(exps)))

    with Parallel(n_jobs = n_procs) as parallel:

        pert = pert_type if pert_type == "rewire" else "creat-dest"

        if pert_type == "creat":
            creat = 2*eps
            dest = 0.
        elif pert_type == "dest":
            dest = 2*eps
            creat = 0.
        else:
            creat = None
            dest = None

        funcs_poly = []
        funcs_sem = []
        for i in range(n_graphs):
            X, Y_poly, Y_sem, Cy_poly, Cy_sem, Cy_samp_poly, Cy_samp_sem, H_poly, H_sem, S, Sn, h = data.gen_data_SEM(N, M, g_params, p_n, eps, K, neg_coefs=neg_coefs, exp_coefs=exp_coefs, sort_h=sort_h, norm_S=norm_S, norm_H=norm_H, pert_type=pert, creat=creat, dest=dest)
                
            funcs_poly.append(delayed(test)(X, Y_poly, Sn, S, H_poly, h, K, Cy_poly, Cy_samp_poly, exps, args))
            funcs_sem.append(delayed(test)(X, Y_sem, Sn, S, H_sem, h, K, Cy_sem, Cy_samp_sem, exps, args))
        
        results_poly = parallel(funcs_poly)
        results_sem = parallel(funcs_sem)

        for i in range(n_graphs):
            err_H[i,0,:], err_S[i,0,:], err_H_coefs[i,0,:], err_h_bar[i,0,:] = results_poly[i]
            err_H[i,1,:], err_S[i,1,:], err_H_coefs[i,1,:], err_h_bar[i,1,:] = results_sem[i]

    return {
        'med_H': np.median(err_H, 0),
        'std_H': np.std(err_H, 0),
        'mean_H': np.mean(err_H, 0),
        'mean_S': np.mean(err_S, 0),
        'med_S': np.median(err_S, 0),
        'std_S': np.std(err_S, 0),
        'med_H_coefs': np.median(err_H_coefs, 0),
        'std_H_coefs': np.std(err_H_coefs, 0),
        'mean_H_coefs': np.mean(err_H_coefs, 0),
        'med_h_bar': np.median(err_h_bar, 0),
        'std_h_bar': np.std(err_h_bar, 0),
        'mean_h_bar': np.mean(err_h_bar, 0)
    }
    