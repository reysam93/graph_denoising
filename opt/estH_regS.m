function [H,S] = estH_regS(X,Y,Sn,Cy,lambda,delta,verbose)
    if nargin < 6
        delta = 0;
    end
    if nargin < 7
        verbose = false;
    end
        
    N = size(X, 1);        
    cvx_begin quiet
        variable S(N,N) symmetric
        variable H(N,N) symmetric
        minimize(vec(Y-H*X)'*vec(Y-H*X) + lambda*norm(vec(S-Sn),1))
        subject to
            diag(S) == 0;
            S >= 0;
            norm(Cy*S-S*Cy,'fro') <= delta;
            norm(Cy*H-H*Cy,'fro') <= delta;
    cvx_end
    if strcmp(cvx_status, 'Failed') || strcmp(cvx_status, 'Infeasible')
       disp(['WARNING: estH_regS cvx status: ' cvx_status]);
       H = [];
       S =[];
    end
    if verbose
        disp(['   - Opt val: ' num2str(cvx_optval)])
    end
end