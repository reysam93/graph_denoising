function [H,S] = estH_denS(X,Y,Sn,Cy,delta,verbose)
    if nargin < 5
        delta = 0;
    end
    if nargin < 6
        verbose = false;
    end
    N = size(X, 1);  
    
    % Denoise S
    cvx_begin quiet
        variable S(N,N) symmetric
        minimize(norm(vec(S-Sn),1))
        subject to
            norm(Cy*S-S*Cy,'fro') <= delta;
            diag(S) == 0;
            S >= 0;
    cvx_end
    
    % Compute H from S
    cvx_begin quiet
        variable H(N,N) symmetric
        minimize(vec(Y-H*X)'*vec(Y-H*X))
        subject to
            norm(S*H-H*S,'fro') <= 0;
    cvx_end
    if strcmp(cvx_status, 'Failed') || strcmp(cvx_status, 'Infeasible')
       disp(['WARNING: estH_denS cvx status: ' cvx_status]);
       H = [];
       S =[];
    end
    if verbose
        disp([' Opt val: ' num2str(cvx_optval)])
    end
end