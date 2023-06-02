import numpy as np
import torch.nn as nn
import torch
from scipy.linalg import khatri_rao

class RobustPolyOpt:
    def __init__(self, S0, K, n_iters_out, n_iters_S, lr, eval_freq):
        self.n_iters_out = n_iters_out
        self.n_iters_S = n_iters_S

        self.lr = lr
        self.eval_freq = eval_freq

        self.model = PolynomialGFModel(S0, K)
        self.opt_S = torch.optim.SGD([self.model.S], lr=self.lr)

        self.loss = nn.MSELoss(reduction='sum')
        

    def calc_loss_S(self, y_hat, y_train, beta=1):
        return self.loss(y_hat, y_train) + beta*torch.sum(self.model.S)


    def gradient_step_S(self, X=None, Y=None, beta=None):
        Y_hat = self.model(X).squeeze()
        loss = self.calc_loss_S(Y_hat, Y, beta)

        self.opt_S.zero_grad()
        loss.backward()
        self.opt_S.step()

        return self.model.S.data


    def steph(self, S, X, Y, true_h):
        norm_h = torch.linalg.norm(true_h)
        S = S.data.numpy()
        X = X.data.numpy()
        y_vec = Y.data.numpy().flatten(order='F')

        eigenval, V = np.linalg.eig(S)
        Psi = np.vander(eigenval, N=self.model.K, increasing=True)
        Theta = khatri_rao(X.T @ V, V) @ Psi
        h = np.linalg.pinv(Theta) @ y_vec
        self.model.h = torch.Tensor(h)
        err_h = torch.linalg.norm(self.model.h - true_h) / norm_h
        return err_h


    def stepS(self, Sn, X, Y, lambd, beta, S_true=None, debug=False):
        errs_S = np.zeros(self.n_iters_S)
        change_S = np.zeros(self.n_iters_S)
        norm_S = torch.linalg.norm(S_true)

        self.model.train()

        lambd *= self.lr
        beta *= self.lr

        for i in range(self.n_iters_S):
            orig_S = self.model.S.data.clone()
            norm_S_orig = torch.linalg.norm(orig_S)
            # S = self.model.S.data.clone()

            # S = self.gradient_step_S(S, beta, X, Y)
            S = self.gradient_step_S(X, Y, beta)


            # Proximal for the distance to S_bar
            idxs_greater = torch.where(S - Sn > lambd)
            idxs_lower = torch.where(S - Sn < -lambd)
            S_prox = Sn.clone()
            S_prox[idxs_greater] = S[idxs_greater] - lambd
            S_prox[idxs_lower] = S[idxs_lower] + lambd
            S = S_prox

            # Projection onto \mathcal{S}
            S = torch.where(S < 0., 0., S)
            S = torch.where(S > 1., 1., S)
            S = (S + S.T) / 2

            errs_S[i] = torch.linalg.norm(S - S_true) / norm_S
            change_S[i] = torch.linalg.norm(S - orig_S) / norm_S_orig

            if debug and (i == 0 or (i+1) % self.eval_freq == 0):
                norm_A =  torch.linalg.norm(S)
                err_S2 = torch.linalg.norm(S/norm_A - S_true/norm_S) 
                print(f'\tEpoch  {i} :  norm(A): {norm_A:.1f}  -  change(S-Sprev): {change_S[i]:.3f}  -  err_S: {errs_S[i]:.3f}  -  err_S (free scale): {err_S2:.3f}')

            self.model.update_S(S)

        return errs_S, change_S
        

    def test_model(self, Sn, X, Y, params, S_true=None, true_h=None,
                   verbose=False, debug_S=False):
        true_h = torch.Tensor(true_h)
        S_true = torch.Tensor(S_true)
        Sn = torch.Tensor(Sn)
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)

        if verbose:
            norm_S = torch.linalg.norm(S_true)
            err_Sn = torch.linalg.norm(S_true - Sn) / norm_S

        lambd, beta = params
        err_h = np.zeros((self.n_iters_out))
        # err_H = np.zeros((self.n_iters_out))
        err_S = np.zeros((self.n_iters_out, self.n_iters_S))
        change_S = np.zeros((self.n_iters_out, self.n_iters_S))

        for i in range(self.n_iters_out):
            err_h[i] = self.steph(self.model.S, X, Y, true_h)
            err_S[i,:], change_S[i,:] = self.stepS(Sn, X, Y, lambd, beta, S_true, debug=debug_S)

            if verbose:
                print(f"Iteration {i+1} DONE - Err h: {err_h[i]:.3f} - Err S: {err_S[i,-1]:.3f} - Err Sn: {err_Sn:.3f}")

        return self.model.h.data.numpy(), self.model.S.data.numpy(), err_h, err_S, change_S


class PolynomialGFModel(nn.Module):
    def __init__(self, S, K):
        super().__init__()
        self.h = torch.ones(K)/K
        self.S = nn.Parameter(torch.Tensor(S))
        self.N = self.S.shape[0]
        self.K = K

    def update_S(self, newS):
        self.S.data = newS

    def forward(self, x):
        Nin, M = x.shape
        assert Nin == self.N

        x_out = self.h[0] * x
        Sx = x
        for k in range(1, self.K):
            Sx = self.S @ Sx
            x_out += self.h[k] * Sx

        return x_out
