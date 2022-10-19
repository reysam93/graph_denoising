# %%
from multiprocessing.sharedctypes import Value
import numpy as np

import cvxpy as cp
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/vtenorio/Codigo/graph_denoising/python/')
import opt
import robustAR_opt
import sev_filters_opt

import time
import argparse
import os

# %%
def error_metrics(Y_hat, Y_test, H=None, H_gt=None, err_deg=False, err_no_mean=False, print_results=True):
    # Normalized error
    if Y_test.ndim == 2:
        if err_no_mean:
            Y_test = Y_test.copy() - Y_test.mean(1)[:,None]
            Y_hat = Y_hat.copy() - Y_hat.mean(1)[:,None]
            err_y = np.mean(((Y_test - Y_hat)**2).sum(0))
            err_y_deg = np.mean(np.abs(Y_test - Y_hat))
        else:
            norm_y = (Y_test**2).sum(0)#np.sqrt((Y_test**2).sum(1))
            err_y = np.mean(((Y_test - Y_hat)**2).sum(0) / norm_y)
            err_y_deg = np.mean(np.abs(Y_test - Y_hat))
    else:
        norm_y = (Y_test**2).sum((1,2))#np.sqrt((Y_test**2).sum(1))
        err_y = np.mean(((Y_test - Y_hat)**2).sum((1,2)) / norm_y)
        err_y_deg = np.mean(np.abs(Y_test - Y_hat))
    if print_results:
        print(f"Error in signal y: {err_y}")
    
    if err_deg or err_no_mean:
        err_y = err_y_deg

    if H is not None and H_gt is not None:
        # Error wrt ground truth filter
        if H.ndim == 3: # Several filters
            norm_H = (H_gt**2).sum((1,2))
            err_H = np.mean(((H - H_gt)**2).sum((1,2)) / norm_H)
        else:
            norm_H = (H_gt**2).sum()
            err_H = ((H - H_gt)**2).sum() / norm_H

        if print_results:
            print(f"Error in filter: {err_H}")
        return err_y, err_H

    return err_y

parser = argparse.ArgumentParser(description='Temperature data experiment - Varying number of steps.')

#parser.add_argument('name', type=str, help='Name to give to the experiment')

parser.add_argument('--norm_data', action='store_true', default=False, help='Whether to normalize the data')
parser.add_argument("--gso_type",  type=str, default="adj", help='Whether to use adjacency (adj), normalized adjacency (norm_adj) or laplacian (lap)')
parser.add_argument("--stationary", action='store_true', default=False, help="Whether to use stationarity")
parser.add_argument("--random_split", action='store_true', default=False, help="Whether to use a random split or continuous in time")
parser.add_argument("--rem_mean", action='store_true', default=False, help="Whether to remove the in-sample mean from the data")
parser.add_argument("--monthly_mean", action='store_true', default=False, help="Whether to remove the monthly moving average from the data")
parser.add_argument("--two_years", action='store_true', default=False, help="Whether to use data from both 2017 and 2018 or only 2018")

parser.add_argument("--tts",  type=float, default=0.5, help='Train test split')
parser.add_argument("--P",  type=int, default=3, help='Order of the autorregressive process for VAR')

parser.add_argument('--do_precs', action='store_true', default=False, help="Whether to include precipitations in computation of several filters")
parser.add_argument('--do_windsps', action='store_true', default=False, help="Whether to include wind speed in computation of several filters")
parser.add_argument('--do_press', action='store_true', default=False, help="Whether to include pressure in computation of several filters")

cli_args = parser.parse_args()


# %%
#data = np.load('data_temp/temperatures2003_mod_knn8_N130.npz')
if cli_args.two_years:
    data = np.load('data_temp/temperatures2017-18_knn5_N19.npz')
else:
    data = np.load('data_temp/temperatures2018_knn5_N31.npz')


# %%
A = data['A_bin']
if cli_args.gso_type == "adj":
    S = A
elif cli_args.gso_type == "lap":
    S = np.diag(np.sum(A, 1)) - A
elif cli_args.gso_type == "norm_adj":
    d = np.sum(A, 1)
    D = np.diag(1/np.sqrt(d))
    S = D @ A @ D
else:
    raise RuntimeError("Not a valid GSO Type")

temp_info = data['temps']

norm_data = False
if norm_data:
    temp_info -= np.mean(temp_info, 1)[:,None]
    temp_info /= np.std(temp_info, 1)[:,None]

if cli_args.norm_data:
    temp_info = temp_info / np.sqrt((temp_info**2).sum(1))[:,None]

if cli_args.monthly_mean:
    n_days = 30

    def moving_average(a, n=2) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    month_mean = np.apply_along_axis(moving_average, 1, temp_info, n_days)
    month_mean_padded = np.pad(month_mean, [(0,0), (n_days-1,0)], 'edge')
    assert month_mean_padded.shape == temp_info.shape, "Something went wrong calculating rolling mean"

    temp_info -= month_mean_padded

P = cli_args.P

tts = cli_args.tts

# For several filters
X1 = data['temps'].copy()
X2 = data['precs'].copy()
X3 = data['windsps'].copy()
X4 = data['press'].copy()

attrs = ["Temperatures"]
comp_data_l = [X1]

if cli_args.do_precs:
    attrs += ["Precipitations"]
    comp_data_l += [X2]

if cli_args.do_windsps:
    attrs += ["Wind Speed"]
    comp_data_l += [X3]

if cli_args.do_press:
    attrs += ["Pressure"]
    comp_data_l += [X4]

comp_data = np.array(comp_data_l)
n_feat, N, _ = comp_data.shape

if cli_args.norm_data:
    comp_data = comp_data / np.sqrt((comp_data**2).sum(2))[:,:,None]

if cli_args.monthly_mean:
    n_days = 30

    def moving_average(a, n=2) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
        
    month_mean = np.apply_along_axis(moving_average, 2, comp_data, n_days)
    month_mean_padded = np.pad(month_mean, [(0,0), (0,0), (n_days-1,0)], 'edge')
    assert month_mean_padded.shape == comp_data.shape, "Something went wrong calculating rolling mean"

    comp_data -= month_mean_padded

# %%
n_steps_vals = [1,2,3,4,5]
#n_steps_vals = [1,5]

do_eye = True
do_ls_gf = True
do_var = True
do_rew = True
do_tls = True
do_scp = False

do_sev = False

n_exps = 3
if do_ls_gf:
    n_exps += 1

if do_eye:
    n_exps += 1

if do_var:
    n_exps += 3
    if do_rew:
        n_exps += 1
    
    if do_ls_gf:
        n_exps += 1

if do_rew:
    n_exps += 1

if do_tls:
    n_exps += 1

if do_scp:
    n_exps += 1

errs_y = np.zeros((n_exps, len(n_steps_vals)))
errs_y_deg = np.zeros((n_exps, len(n_steps_vals)))
errs_y_0mean = np.zeros((n_exps, len(n_steps_vals)))
errs_y_all = np.zeros((n_exps, len(n_steps_vals)))
errs_H = np.zeros((n_exps, len(n_steps_vals)))

n_exps_sev = 4
if do_ls_gf:
    n_exps_sev += 1

if do_eye:
    n_exps_sev += 1

if do_rew:
    n_exps_sev += 2

errs_y_sev = np.zeros((n_exps_sev, len(n_steps_vals)))
errs_H_sev = np.zeros((n_exps_sev, len(n_steps_vals)))

# %%
args = [0.4, 0.008, 0.0005, 3.]
args_no_st = [0.03, 0.005, 0., 3.5]
args_rew = [0.2, 0.1, 0.001, 0.0001, 1.1]
args_rew_no_st = [0.1, 0.08, 0., 5e-4, 1.]

args_sev = [0.2, 1e-3, 0.02, 1.6]
args_sev_no_st = [0.4, 1e-3, 0., 1.5]
args_sev_rew = [0.07, 1e-3, 0.1, 5e-4, 3.]
args_sev_rew_no_st = [0.02, 1e-4, 0., 5e-4, 2.]

if cli_args.stationary:
    print("Stationarity: Picking stationary parameters")
    params = args
    params_rew = args_rew
    params_sev = args_sev
    params_sev_rew = args_sev_rew
else:
    print("Stationarity: Not using Covariance")
    params = args_no_st
    params_rew = args_rew_no_st
    params_sev = args_sev_no_st
    params_sev_rew = args_sev_rew_no_st

start_time = time.time()

for i, n_steps in enumerate(n_steps_vals):
    print(f"Experiment {i+1}/{len(n_steps_vals)}: Number of steps: {n_steps} - ", flush=True, end="")

    Y = temp_info[:,n_steps:].copy()
    X = temp_info[:,:-n_steps].copy()
    N, N_dates = X.shape

    Xs_var = np.array([temp_info[:,p:-(n_steps-1)-P+p] for p in range(P)])
    Y_var = temp_info[:,P+(n_steps-1):]
    N_dates_var = Y_var.shape[1]

    Ys = comp_data[:,:,n_steps:].copy()
    Xs = comp_data[:,:,:-n_steps].copy()
    Xs.shape, Ys.shape

    N_dates_sev = Xs.shape[1]

    if cli_args.random_split:
        idxs = np.random.permutation(N_dates)
    else:
        idxs = np.arange(N_dates)

    idxs_train = idxs[:int(tts*N_dates)]
    idxs_test = idxs[int(tts*N_dates):]

    X_train = X[:,idxs_train]
    Y_train = Y[:,idxs_train]
    X_test = X[:,idxs_test]
    Y_test = Y[:,idxs_test]
    X_train.shape, Y_train.shape, N_dates

    if cli_args.rem_mean:
        X_train -= X_train.mean(1)[:,None]
        X_test -= X_train.mean(1)[:,None]
        Y_train -= X_train.mean(1)[:,None]
        Y_test -= X_train.mean(1)[:,None]

    j = 0

    # %% [markdown]
    # # Models

    # %%
    models = []
    models_sev = []

    # %%
    # Ground truth obtained using Least squares over all samples
    H = cp.Variable((N,N))
    obj = cp.sum_squares(Y_test - H @ X_test)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    H_gt = H.value

    # %%
    y_ls_perf = H_gt @ X_test
    errs_y[j,i], errs_H[j,i] = error_metrics(y_ls_perf, Y_test, H_gt, H_gt, print_results=False)
    errs_y_deg[j,i] = error_metrics(y_ls_perf, Y_test, err_deg=True, print_results=False)
    errs_y_0mean[j,i] = error_metrics(y_ls_perf.copy(), Y_test.copy(), err_no_mean=True, print_results=False)

    H = cp.Variable((N,N))
    obj = cp.sum_squares(Y - H @ X)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    H_gt_all = H.value
    y_ls_perf_all = H_gt_all @ X
    errs_y_all[j,i] = error_metrics(y_ls_perf_all, Y, print_results=False)
    models.append("LS-Perfect (LB)")
    j += 1

    # %% [markdown]
    # ## Least squares

    # %%
    # Least squares
    H = cp.Variable((N,N))
    obj = cp.sum_squares(Y_train - H @ X_train)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    H_ls = H.value

    # %%
    Y_hat_ls = H_ls @ X_test
    errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_ls, Y_test, H_ls, H_gt, print_results=False)
    errs_y_deg[j,i] = error_metrics(Y_hat_ls, Y_test, err_deg=True, print_results=False)
    errs_y_0mean[j,i] = error_metrics(Y_hat_ls, Y_test, err_no_mean=True, print_results=False)
    Y_hat_ls_all = H_ls @ X
    errs_y_all[j,i] = error_metrics(Y_hat_ls_all, Y, print_results=False)
    models.append("Least Squares")
    j += 1

    if do_ls_gf:
        # %%
        # Least squares postulating it is a graph filter
        K = 5
        Spow = np.array([np.linalg.matrix_power(S, k) for k in range(K)])
        h = cp.Variable(K)
        obj = cp.sum_squares(Y_train - cp.sum([h[k] * Spow[k,:,:] for k in range(K)]) @ X_train)
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()
        h = h.value
        H_ls_gf = np.sum([h[k] * Spow[k,:,:] for k in range(K)], 0)

        # %%
        Y_hat_ls_gf = H_ls_gf @ X_test
        errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_ls_gf, Y_test, H_ls_gf, H_gt, print_results=False)
        errs_y_deg[j,i] = error_metrics(Y_hat_ls_gf, Y_test, err_deg=True, print_results=False)
        errs_y_0mean[j,i] = error_metrics(Y_hat_ls_gf, Y_test, err_no_mean=True, print_results=False)
        Y_hat_ls_gf_all = H_ls_gf @ X
        errs_y_all[j,i] = error_metrics(Y_hat_ls_gf_all, Y, print_results=False)
        models.append("Least Squares-GF")
        j += 1

    if do_eye:
        # Identity - copying the temperature from the previous day
        H_eye = np.eye(N)
        Y_eye = H_eye @ X_test
        errs_y[j,i], errs_H[j,i] = error_metrics(Y_eye, Y_test, H_eye, H_gt, print_results=False)
        errs_y_deg[j,i] = error_metrics(Y_eye, Y_test, err_deg=True, print_results=False)
        errs_y_0mean[j,i] = error_metrics(Y_eye, Y_test, err_no_mean=True, print_results=False)
        Y_eye_all = H_eye @ X
        errs_y_all[j,i] = error_metrics(Y_eye_all, Y, print_results=False)
        models.append("Copy Prev Day")
        j += 1

    # %% [markdown]
    # ## Iterative (robust) algorithms

    # %%
    X_norm = (X_train.T - np.mean(X_train.T, axis=0))#/np.std(X_train.T, axis=0)

    C = np.cov(X_norm.T)

    # %%
    iter, H_iter, S_iter = opt.estH_iter(X, Y, S, C, params)

    # %%
    Y_hat_iter = H_iter @ X_test
    errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_iter, Y_test, H_iter, H_gt, print_results=False)
    errs_y_deg[j,i] = error_metrics(Y_hat_iter, Y_test, err_deg=True, print_results=False)
    errs_y_0mean[j,i] = error_metrics(Y_hat_iter, Y_test, err_no_mean=True, print_results=False)
    Y_hat_iter_all = H_iter @ X
    errs_y_all[j,i] = error_metrics(Y_hat_iter_all, Y, print_results=False)
    models.append("RGFI")
    j += 1

    if do_rew:
        # %%
        iter, H_iter_rew, S_iter_rew = opt.estH_iter_rew(X, Y, S, C, params_rew)

        # %%
        Y_hat_iter_rew = H_iter_rew @ X_test
        errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_iter_rew, Y_test, H_iter_rew, H_gt, print_results=False)
        errs_y_deg[j,i] = error_metrics(Y_hat_iter_rew, Y_test, err_deg=True, print_results=False)
        errs_y_0mean[j,i] = error_metrics(Y_hat_iter_rew, Y_test, err_no_mean=True, print_results=False)
        Y_hat_iter_rew_all = H_iter_rew @ X
        errs_y_all[j,i] = error_metrics(Y_hat_iter_rew_all, Y, print_results=False)
        models.append("RGFI-REW")
        j += 1

    if do_tls:
        args_tls = [0.07, 0.4]
        # %%
        iter, H_tls, S_tls = opt.estH_tls_sem(X, Y, S, C, args_tls)

        # %%
        Y_hat_tls = H_tls @ X_test
        errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_tls, Y_test, H_tls, H_gt, print_results=False)
        errs_y_deg[j,i] = error_metrics(Y_hat_tls, Y_test, err_deg=True, print_results=False)
        errs_y_0mean[j,i] = error_metrics(Y_hat_tls, Y_test, err_no_mean=True, print_results=False)
        Y_hat_tls_all = H_tls @ X
        errs_y_all[j,i] = error_metrics(Y_hat_tls_all, Y, print_results=False)
        models.append("TLS-SEM")
        j += 1

    if do_scp:
        args_scp = [1.]
        # %%
        iter, H_scp, S_scp = opt.estH_llsscp(X, Y, S, C, args_scp)

        # %%
        Y_hat_scp = H_scp @ X_test
        errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_scp, Y_test, H_scp, H_gt, print_results=False)
        errs_y_deg[j,i] = error_metrics(Y_hat_scp, Y_test, err_deg=True, print_results=False)
        errs_y_0mean[j,i] = error_metrics(Y_hat_scp, Y_test, err_no_mean=True, print_results=False)
        Y_hat_scp_all = H_scp @ X
        errs_y_all[j,i] = error_metrics(Y_hat_scp_all, Y, print_results=False)
        models.append("SCP")
        j += 1

    if do_var:

        # %% [markdown]
        # # Multiple filters ARMA model

        # %%
        idxs_mult = idxs.copy() # Using same split as in the previous cases, but needs to be adapted
        for p in range(N_dates-P, N_dates): # Deleting invalid occurrences for this case
            idxs_mult = idxs_mult[idxs_mult != p]
        idxs_train = idxs_mult[:int(tts*N_dates_var)]
        idxs_test = idxs_mult[int(tts*N_dates_var):]
        Xs_var_train = Xs_var[:,:,idxs_train]
        Xs_var_test = Xs_var[:,:,idxs_test]
        Y_var_train = Y_var[:,idxs_train]
        Y_var_test = Y_var[:,idxs_test]

        if cli_args.rem_mean:
            Xs_var_train -= Xs_var_train.mean(2)[:,:,None]
            Xs_var_test -= Xs_var_train.mean(2)[:,:,None]
            Y_var_train -= Xs_var_train.mean(2)[-1,:][:,None]
            Y_var_test -= Xs_var_train.mean(2)[-1,:][:,None]

        # %%
        Hs = [cp.Variable((N, N)) for _ in range(P)]
        obj = cp.sum_squares(Y_var_test - cp.sum([Hs[p] @ Xs_var_test[p,:,:] for p in range(P)]))
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()
        Hs_gt = np.array([Hs[p].value for p in range(P)])

        # %%
        Y_hat_perf = np.sum(Hs_gt @ Xs_var_test, 0)
        errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_perf, Y_var_test, Hs_gt, Hs_gt, print_results=False)
        errs_y_deg[j,i] = error_metrics(Y_hat_perf, Y_var_test, err_deg=True, print_results=False)
        errs_y_0mean[j,i] = error_metrics(Y_hat_perf, Y_var_test, err_no_mean=True, print_results=False)

        Hs = [cp.Variable((N, N)) for _ in range(P)]
        obj = cp.sum_squares(Y_var - cp.sum([Hs[p] @ Xs_var[p,:,:] for p in range(P)]))
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()
        Hs_gt_all = np.array([Hs[p].value for p in range(P)])
        Y_hat_perf_all = np.sum(Hs_gt_all @ Xs_var, 0)
        errs_y_all[j,i] = error_metrics(Y_hat_perf_all, Y_var, print_results=False)
        models.append("VAR-LS-Perfect")
        j += 1

        # %% [markdown]
        # ## Least squares

        # %%
        Hs = [cp.Variable((N, N)) for _ in range(P)]
        obj = cp.sum_squares(Y_var_train - cp.sum([Hs[p] @ Xs_var_train[p,:,:] for p in range(P)]))
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()
        Hs_ls = np.array([Hs[p].value for p in range(P)])

        # %%
        Y_hat_var_ls = np.sum(Hs_ls @ Xs_var_test, 0)
        errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_var_ls, Y_var_test, Hs_ls, Hs_gt, print_results=False)
        errs_y_deg[j,i] = error_metrics(Y_hat_var_ls, Y_var_test, err_deg=True, print_results=False)
        errs_y_0mean[j,i] = error_metrics(Y_hat_var_ls, Y_var_test, err_no_mean=True, print_results=False)
        Y_hat_var_ls_all = np.sum(Hs_ls @ Xs_var, 0)
        errs_y_all[j,i] = error_metrics(Y_hat_var_ls_all, Y_var, print_results=False)
        models.append("VAR-LS")
        j += 1

        if do_ls_gf:
            # %%
            # Least squares postulating it is a graph filter
            K = 5
            hs = [cp.Variable(K) for _ in range(P)]
            Hs = [cp.sum([hs[p][k] * Spow[k,:,:] for k in range(K)]) for p in range(P)]
            obj = cp.sum_squares(Y_var_train - cp.sum([Hs[p] @ Xs_var_train[p,:,:] for p in range(P)]))
            prob = cp.Problem(cp.Minimize(obj))
            prob.solve()
            hs_ls = np.array([hs[p].value for p in range(P)])
            H_var_ls_gf = np.array([np.sum([hs_ls[p,k] * Spow[k,:,:] for k in range(5)], 0) for p in range(P)])

            # %%
            Y_hat_var_ls_gf = np.sum(H_var_ls_gf @ Xs_var_test, 0)
            errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_var_ls_gf, Y_var_test, H_var_ls_gf, Hs_gt, print_results=False)
            errs_y_deg[j,i] = error_metrics(Y_hat_var_ls_gf, Y_var_test, err_deg=True, print_results=False)
            errs_y_0mean[j,i] = error_metrics(Y_hat_var_ls_gf, Y_var_test, err_no_mean=True, print_results=False)
            Y_hat_var_ls_gf_all = np.sum(H_var_ls_gf @ Xs_var, 0)
            errs_y_all[j,i] = error_metrics(Y_hat_var_ls_gf_all, Y_var, print_results=False)
            models.append("VAR-LS-GF")
            j += 1

        # %% [markdown]
        # ## Iterative (robust) algorithms

        # %%
        iter, Hs_iter, Ss_iter = robustAR_opt.estHs_iter(Xs_var_train, Y_var_train, S, C, params_sev)

        # %%
        Y_hat_var_iter = np.sum(Hs_iter @ Xs_var_test, 0)
        errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_var_iter, Y_var_test, Hs_iter, Hs_gt, print_results=False)
        errs_y_deg[j,i] = error_metrics(Y_hat_var_iter, Y_var_test, err_deg=True, print_results=False)
        errs_y_0mean[j,i] = error_metrics(Y_hat_var_iter, Y_var_test, err_no_mean=True, print_results=False)
        Y_hat_var_iter_all = np.sum(Hs_iter @ Xs_var, 0)
        errs_y_all[j,i] = error_metrics(Y_hat_var_iter_all, Y_var, print_results=False)
        models.append("VAR-RGFI")
        j += 1

        if do_rew:
            # %%
            iter, Hs_iter_rew, Ss_iter_rew = robustAR_opt.estHs_iter_rew(Xs_var_train, Y_var_train, S, C, params_sev_rew)

            # %%
            Y_hat_var_iter_rew = np.sum(Hs_iter_rew @ Xs_var_test, 0)
            errs_y[j,i], errs_H[j,i] = error_metrics(Y_hat_var_iter_rew, Y_var_test, Hs_iter_rew, Hs_gt, print_results=False)
            errs_y_deg[j,i] = error_metrics(Y_hat_var_iter_rew, Y_var_test, err_deg=True, print_results=False)
            errs_y_0mean[j,i] = error_metrics(Y_hat_var_iter_rew, Y_var_test, err_no_mean=True, print_results=False)
            Y_hat_var_iter_rew_all = np.sum(Hs_iter_rew @ Xs_var, 0)
            errs_y_all[j,i] = error_metrics(Y_hat_var_iter_rew_all, Y_var, print_results=False)
            models.append("VAR-RGFI-REW")
            j += 1

    assert j == n_exps, "Number of experiments wrongly calculated"

    if do_sev:
        j = 0
        print("Several filters - ", flush=True, end="")

        Xs_train = Xs[:,:,idxs_train]
        Ys_train = Ys[:,:,idxs_train]
        Xs_test = Xs[:,:,idxs_test]
        Ys_test = Ys[:,:,idxs_test]
        Xs_train.shape, Ys_train.shape, Xs_test.shape, N_dates

        if cli_args.rem_mean:
            Xs_train -= Xs_train.mean(2)[:,:,None]
            Ys_train -= Xs_train.mean(2)[:,:,None]
            Xs_test -= Xs_train.mean(2)[:,:,None]
            Ys_test -= Xs_train.mean(2)[:,:,None]

        # For the case of several filters
        # Ground truth
        Hs_gt = []
        for t in range(n_feat):
            H = cp.Variable((N,N))
            obj = cp.sum_squares(Ys_test[t,:,:] - H @ Xs_test[t,:,:])
            prob = cp.Problem(cp.Minimize(obj))
            prob.solve()
            Hs_gt.append(H.value)
        Hs_gt = np.array(Hs_gt)

        Ys_ls_perf = Hs_gt @ Xs_test
        errs_y_sev[j,i], errs_H_sev[j,i] = error_metrics(Ys_ls_perf, Ys_test, Hs_gt, Hs_gt, print_results=False)
        models_sev.append("LS-Perfect (LB)")
        j += 1

        if do_eye:
            # Identity - copying the temperature from the previous day
            Hs_eye = np.array([np.eye(N) for _ in range(n_feat)])
            Ys_eye = Hs_eye @ Xs_test
            errs_y_sev[j,i], errs_H_sev[j,i]  = error_metrics(Ys_eye, Ys_test, Hs_eye, Hs_gt, print_results=False)
            models_sev.append("Copy Prev Day")
            j += 1

        # Least squares
        Hs_ls = []
        for t in range(n_feat):
            H = cp.Variable((N,N))
            obj = cp.sum_squares(Ys_train[t,:,:] - H @ Xs_train[t,:,:])
            prob = cp.Problem(cp.Minimize(obj))
            try:
                prob.solve()
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    Hs_ls.append(H.value)
                else:
                    Hs_ls.append(np.eye(N))
            except cp.SolverError:
                print("There has been a solver error in feat " + str(i))
                Hs_ls.append(np.eye(N))
            
        Hs_ls = np.array(Hs_ls)

        try:
            Ys_ls = Hs_ls @ Xs_test
        except ValueError:
            print(Hs_ls.shape)
            print(Xs_test.shape)
            raise ValueError
        errs_y_sev[j,i], errs_H_sev[j,i] = error_metrics(Ys_ls, Ys_test, Hs_ls, Hs_gt, print_results=False)
        models_sev.append("Least Squares")
        j += 1

        if do_ls_gf:
            # Least squares postulating it is a graph filter
            K = 5
            Hs_ls_gf = []
            for t in range(n_feat):
                h = cp.Variable(K)
                obj = cp.sum_squares(Ys_train[t,:,:] - cp.sum([h[k] * Spow[k,:,:] for k in range(K)]) @ Xs_train[t,:,:])
                prob = cp.Problem(cp.Minimize(obj))
                prob.solve()
                h = h.value
                Hs_ls_gf.append(np.sum([h[k] * Spow[k,:,:] for k in range(K)], 0))
            Hs_ls_gf = np.array(Hs_ls_gf)

            Ys_ls_gf = Hs_ls_gf @ Xs_test
            errs_y_sev[j,i], errs_H_sev[j,i] = error_metrics(Ys_ls_gf, Ys_test, Hs_ls_gf, Hs_gt, print_results=False)
            models_sev.append("Least Squares-GF")
            j += 1

        meanX = comp_data.mean(2)
        stdX = comp_data.std(2)
        Cs = np.zeros((n_feat, N, N))
        for t in range(n_feat):
            X_norm = (comp_data[t,:,:].T - meanX[t,:]) / stdX[t,:]

            X_norm = np.where(np.isnan(X_norm), comp_data[t,:,:].T, X_norm)

            Cs[t,:,:] = np.cov(X_norm.T)

        # Independent RGFI
        Hs_iter = []
        Hs_iter_rew = []
        for t in range(n_feat):
            _, H, _ = opt.estH_iter(Xs[t,:,:], Ys[t,:,:], S, Cs[t,:,:], params)
            Hs_iter.append(H)
            if do_rew:
                _, H, _ = opt.estH_iter_rew(Xs[t,:,:], Ys[t,:,:], S, Cs[t,:,:], params_rew)
                Hs_iter_rew.append(H)
        Hs_iter = np.array(Hs_iter)
        Hs_iter_rew = np.array(Hs_iter_rew)

        Ys_iter = Hs_iter @ Xs_test
        errs_y_sev[j,i], errs_H_sev[j,i] = error_metrics(Ys_iter, Ys_test, Hs_iter, Hs_gt, print_results=False)
        models_sev.append("RGFI")
        j += 1

        if do_rew:
            Ys_iter_rew = Hs_iter_rew @ Xs_test
            errs_y_sev[j,i], errs_H_sev[j,i] = error_metrics(Ys_iter_rew, Ys_test, Hs_iter_rew, Hs_gt, print_results=False)
            models_sev.append("RGFI-REW")
            j += 1

        iter, Hs_iter_j, S_iter_j = sev_filters_opt.estHs_iter(Xs_train, Ys_train, S, Cs, params_sev)

        Ys_iter_j = Hs_iter_j @ Xs_test
        errs_y_sev[j,i], errs_H_sev[j,i] = error_metrics(Ys_iter_j, Ys_test, Hs_iter_j, Hs_gt, print_results=False)
        models_sev.append("RGFI-J")
        j += 1


        if do_rew:
            iter, Hs_iter_rew_j, S_iter_rew_j = sev_filters_opt.estHs_iter_rew(Xs_train, Ys_train, S, Cs, params_sev_rew)

            Ys_iter_rew_j = Hs_iter_rew_j @ Xs_test
            errs_y_sev[j,i], errs_H_sev[j,i] = error_metrics(Ys_iter_rew_j, Ys_test, Hs_iter_rew_j, Hs_gt, print_results=False)
            models_sev.append("RGFI-REW-J")
            j += 1

        assert j == n_exps_sev, "Number of experiments wrongly calculated in several filters"
    print(f"DONE - {time.strftime('%H:%M')}")


print('--- {} minutes ---'.format((time.time()-start_time)/60))

if not os.path.exists("results"):
    os.mkdir("results")

today = time.strftime('%Y%m%d')
path = f"results/{today}"
if not os.path.exists(path):
    os.mkdir(path)

name = f"Steps-N{N}-"
if cli_args.norm_data:
    name += "Norm-"

if cli_args.do_precs:
    name += "Precs-"
if cli_args.do_windsps:
    name += "Wind-"
if cli_args.do_press:
    name += "Press-"

if cli_args.stationary:
    name += "Stat-"

if cli_args.random_split:
    name += "RandSplit-"
else:
    name += "ContSplit-"

if cli_args.rem_mean:
    name += "0Mean-"

if cli_args.monthly_mean:
    name += "MonthMA-"

name += f"P{P}-{tts}tts"

#name += cli_args.gso_type

path += "/" + name

if not os.path.exists(path):
    os.mkdir(path)

## Saving results
np.savez(f"{path}/results", errs_y=errs_y, errs_y_deg=errs_y_deg, errs_y_all=errs_y_all, errs_y_0mean=errs_y_0mean, errs_H=errs_H, models=models,
    errs_y_sev=errs_y_sev, errs_H_sev=errs_H_sev, models_sev=models_sev,
    norm_data=cli_args.norm_data, P=P, attrs=attrs, gso_type=cli_args.gso_type, stationary=cli_args.stationary,
    monthly_mean=cli_args.monthly_mean, tts=tts
)

## Errors in y
f, ax = plt.subplots(figsize=(12,8))

for i,m in enumerate(models):
    ax.semilogy(n_steps_vals, errs_y[i,:], label=m)

ax.legend(fontsize=14)
ax.set_title("Error measured in y", fontsize=16)
ax.set_xlabel("Number of steps in the future", fontsize=12)
ax.set_ylabel("Normalized Error", fontsize=12)

f.savefig(f"{path}/err_y.png")

## Error in filter
f, ax = plt.subplots(figsize=(12,8))

for i,m in enumerate(models):
    ax.semilogy(n_steps_vals, errs_H[i,:], label=m)

ax.legend(fontsize=14)
ax.set_title("Error measured in H", fontsize=16)
ax.set_xlabel("Number of steps in the future", fontsize=12)
ax.set_ylabel("Normalized Error", fontsize=12)

f.savefig(f"{path}/err_H.png")

if do_sev:
    ## Errors in y
    f, ax = plt.subplots(figsize=(12,8))

    for i,m in enumerate(models_sev):
        ax.semilogy(n_steps_vals, errs_y_sev[i,:], label=m)

    ax.legend(fontsize=14)
    ax.set_title("Error measured in y", fontsize=16)
    ax.set_xlabel("Number of steps in the future", fontsize=12)
    ax.set_ylabel("Normalized Error", fontsize=12)

    f.savefig(f"{path}/err_y_sev.png")

    ## Error in filter
    f, ax = plt.subplots(figsize=(12,8))

    for i,m in enumerate(models_sev):
        ax.semilogy(n_steps_vals, errs_H_sev[i,:], label=m)

    ax.legend(fontsize=14)
    ax.set_title("Error measured in H", fontsize=16)
    ax.set_xlabel("Number of steps in the future", fontsize=12)
    ax.set_ylabel("Normalized Error", fontsize=12)

    f.savefig(f"{path}/err_H_sev.png")