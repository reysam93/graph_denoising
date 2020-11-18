function [H,S] = estH_tls_sem_noise(X,Yn,Sn,lambda1,lambda2,max_iters,verbose)
    if nargin < 6
        max_iters = 10;
    end
    if nargin < 7
        verbose = false;
    end
    
    N = size(Yn,1);   
    Y = Yn;
    err = zeros(max_iters,1);
    for i=1:max_iters
        cvx_begin quiet
            variable Delta(N,N) symmetric
            minimize(vec(Y-(Sn-Delta)*Y-X)'*vec(Y-(Sn-Delta)*Y-X)...
                + lambda1*norm(vec(Delta),1))
            subject to
                diag(Delta)==0;
        cvx_end
        
        if strcmp(cvx_status, 'Failed') || strcmp(cvx_status, 'Infeasible')
            disp(['WARNING: TLS SEM: ' cvx_status]);
            H = [];
            S =[];
            err(i) = 1;
            continue
        else
            S = Sn-Delta;
            S_tilde = eye(N)-S;
            H = inv(eye(N)-S);
        end
        
        % Analytical solution
        Y = (S_tilde^2+lambda2*eye(N))\(S_tilde*X+lambda2*Yn);
        
        err(i) = vec(Y-(Sn-Delta)*Y-X)'*vec(Y-(Sn-Delta)*Y-X)+...
            lambda1*norm(vec(Delta),1)+lambda2*vec(Y-Yn)'*vec(Y-Yn);
        if verbose
            disp(['   Iter: ' num2str(i) ' Opt val: ' num2str(err(i))])
        end
        
        if i>1 && abs(err(i)-err(i-1))<1e-3
            break
        end
    end
end