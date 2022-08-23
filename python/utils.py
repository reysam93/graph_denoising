import sys
sys.path.append('..')

import data
import opt

import numpy as np

from joblib import Parallel, delayed, cpu_count
import json

import pandas as pd

def read_from_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content

def results_to_csv(results, x_ax, col_names, metric, csv_path):
    df = pd.DataFrame(columns=[x_ax, *col_names])
    df[x_ax] = list(results.keys())
    for i, col in enumerate(col_names):
        if isinstance(list(results.values())[0], dict):
            df[col] = [r[metric][i] for r in results.values()]
        else:
            df[col] = [r[i] for r in results.values()]
    
    df.to_csv(csv_path, sep=';', index=False, index_label=False)

def test(X, Y, Sn, S, H, h, K, Cy, Cy_samp, exps, args, S_for_coefs="binarized"):
    err_H = np.zeros((len(exps)))
    err_S = np.zeros((len(exps)))
    err_h_bar = np.zeros((len(exps)))
    err_H_coefs = np.zeros((len(exps)))

    norm_h = (H**2).sum()
    norm_A = (S**2).sum()
    h_true, h_bar_true = data.obtain_filter_coefs(S, H, K, True)
    norm_h_bar = (h_bar_true**2).sum()
    #assert np.allclose(h, h_true) # Care with normalization
    import warnings
    warnings.filterwarnings('ignore')

    for i, exp in enumerate(exps):

        params = args[exp["cy"]].get(exp["func"], [])

        if exp["cy"] == "real":
            Cy_exp = Cy
        elif exp["cy"] == "samp":
            Cy_exp = Cy_samp
        elif exp["cy"] == "non-st":
            #assert 0. in params, "Delta is not 0" # Warning - this does not check that delta is 0, is just making sure SOME parameter is 0
            Cy_exp = Cy
        else:
            raise NotImplementedError("Choose either real or samp covariance")

        if "iter" in exp["func"] or "tls" in exp["func"]:
            kwargs = {"H_true": H, "S_true": S}
        elif 'llsscp' in exp["func"]:
            kwargs = {"H_true": H, "S_true": S, 'K': K}
        else:
            kwargs = {}

        iter, H_est, S_est = getattr(opt, exp["func"])(X, Y, Sn, Cy_exp, params, **kwargs)

        err_H[i] = ((H - H_est)**2).sum()/norm_h
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

        h_est, h_bar_est = data.obtain_filter_coefs(S_est_coefs, H_est, K, return_h_bar=True)
        h_est = h_est / (h_est**2).sum()
        h_bar_est = h_bar_est / (h_bar_est**2).sum()
        err_h_bar[i] = ((h_bar_est - h_bar_true/norm_h_bar)**2).sum()
        err_H_coefs[i] = ((h - h_est)**2).mean()
        #print(f"Function {f} took {iter} iterations on covariance {cy_use}")
        
    return err_H, err_S, err_H_coefs, err_h_bar


def objective(p_n, M, K, eps, exps, args, n_graphs, N, g_params, neg_coefs=False, exp_coefs=True, sort_h=False, norm_S=False, norm_H=False, pert_type="rewire", sel_ratio=1, sel_node_idx=0, n_procs=cpu_count()):

    err_H = np.zeros((n_graphs, len(exps)))
    err_S = np.zeros((n_graphs, len(exps)))
    err_H_coefs = np.zeros((n_graphs, len(exps)))
    err_h_bar = np.zeros((n_graphs, len(exps)))

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

        funcs = []
        for i in range(n_graphs):
            X, Y, Cy, Cy_samp, H, S, Sn, h = data.gen_data(N, M, g_params, p_n, eps, K, neg_coefs=neg_coefs, exp_coefs=exp_coefs, sort_h=sort_h, norm_S=norm_S, norm_H=norm_H, pert_type=pert, creat=creat, dest=dest, sel_ratio=sel_ratio, sel_node_idx=sel_node_idx)
                
            funcs.append(delayed(test)(X, Y, Sn, S, H, h, K, Cy, Cy_samp, exps, args))
        
        results = parallel(funcs)

        for i in range(n_graphs):
            err_H[i,:], err_S[i,:], err_H_coefs[i,:], err_h_bar[i,:] = results[i]

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


def objective_sem(p_n, M, K, eps, exps, args, n_graphs, N, g_params, neg_coefs=False, exp_coefs=True, sort_h=False, norm_S=False, norm_H=False, pert_type="rewire", n_procs=cpu_count()):
    err_H = np.zeros((n_graphs, 2, len(exps)))
    err_S = np.zeros((n_graphs, 2, len(exps)))
    err_H_coefs = np.zeros((n_graphs, 2, len(exps)))
    err_h_bar = np.zeros((n_graphs, 2, len(exps)))

    with Parallel(n_jobs = n_procs, backend='multiprocessing') as parallel:

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
    