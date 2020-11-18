function [H,S] = estH_non_st(X,Y,Sn,lambda,gamma,max_iters,inc_gamma,verbose)
    if nargin < 6
        max_iters = 20;
    end
    if nargin < 7
        inc_gamma = true;
    end
    if nargin < 8
        verbose = false;
    end
    N = size(X, 1); 
    M = size(X, 2); 
    H_prev = [];
    S_prev = Sn;
    S = Sn;
    
    
    prev_err1 = false;
    prev_err2 = false;
    err = zeros(max_iters,1);
    err_Y_HX = zeros(max_iters,1);
    err_HS_SH = zeros(max_iters,1);
    err_HS_SH_2 = zeros(max_iters*2,1);
    err_S_Sn = zeros(max_iters,1);
    for i=1:max_iters
        cvx_begin quiet
            variable H(N,N) symmetric
            minimize(vec(Y-H*X)'*vec(Y-H*X)+gamma*vec(H*S-S*H)'*vec(H*S-S*H))
        cvx_end
        
        if strcmp(cvx_status, 'Failed') || strcmp(cvx_status, 'Infeasible')
           disp(['WARNING: estH_non_st (1) iter ' num2str(i)...
               ': cvx status: ' cvx_status ' norm H: '  num2str(norm(H,'fro'))]);
           H = H_prev;
           if prev_err1
               break
           else
               prev_err1 = true;
               continue
           end
        else
            H_prev = H;
            prev_err1 = false;
        end
%         I = eye(N);
%         Z = gamma*(kron(S*S',I)+kron(I,S'*S)-kron(S',S')-kron(S,S));
%         H = (kron(X*X',I)+Z)\kron(X,I)*vec(Y);
%         H = reshape(H,[N,N]);
        
        err_HS_SH_2(i*2-1) = vec(H*S-S*H)'*vec(H*S-S*H);
        
        cvx_begin quiet
            variable S(N,N) symmetric
            minimize(lambda*norm(vec(S-Sn),1)+gamma*vec(H*S-S*H)'*vec(H*S-S*H))
            subject to
                S >= 0;
                diag(S) == 0;
        cvx_end
        
        S_nan = sum(sum(isnan(S)));
        if strcmp(cvx_status, 'Failed') || strcmp(cvx_status, 'Infeasible') || S_nan > 0
           disp(['WARNING: estH_non_st (1) iter ' num2str(i)...
               ': cvx status: ' cvx_status ' - Nans: ' num2str(S_nan)]);
           S = S_prev;
           if prev_err2
               break
           else
               prev_err2 = true;
               continue
           end
        else
            S_prev = S;
            prev_err2 = false;
        end
        
        err_Y_HX(i) = vec(Y-H*X)'*vec(Y-H*X);
        err_HS_SH(i) = vec(H*S-S*H)'*vec(H*S-S*H);
        err_S_Sn(i) = norm(vec(S-Sn),1);
        err_HS_SH_2(i*2) = vec(H*S-S*H)'*vec(H*S-S*H);
        
        
        err(i) = vec(Y-H*X)'*vec(Y-H*X)+lambda*norm(vec(S-Sn),1)+gamma*vec(H*S-S*H)'*vec(H*S-S*H);
        if verbose
            disp([' Iter ' num2str(i) ': Opt val: ' num2str(err(i))...
                ': |Y-HX|=' num2str(vec(Y-H*X)'*vec(Y-H*X)) ' - |HS-SH|='...
                num2str(vec(H*S-S*H)'*vec(H*S-S*H)) ' - |S-Sn|='...
                num2str(norm(vec(S-Sn),1))])
        end
        if i>1 && abs(err(i)-err(i-1))<1e-2
            break
        end
        if inc_gamma
            gamma = gamma*1.1;
        end
    end
end