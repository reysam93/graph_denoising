function [H_t, S] = estH_time_varying(X_t,Y_t, Sn_t, Cy, Cy_samp, param)
%ESTH_ Summary of this function goes here
%   Detailed explanation goes here

    T = param{1, 1};
    max_iters = param{1, 2};
    lambda_t = param{1, 3};
    gamma = param{1, 4};
    beta = param{1, 5};
    alpha = param{1, 6};
    gamma_inc = param{1, 7};
    single_S_observation = param{1,8};
    stationnarity = param{1,9};
    delta = param{1,10};

    
    s = size(X_t);
    N = s(1);
    
    S = Sn_t(:,:,1);
    
    backupS = S;
    backupH_t = zeros(N,N,T);
    for j=1:max_iters
        
        gamma = gamma * gamma_inc;
        
        disp(['Iteration ' num2str(j)])
        
        cvx_begin quiet

            variable H_t(N, N, T)

            goal = 0;

            for i=1:T
                goal = goal + (vec(Y_t(:,:,i)-H_t(:,:,i)*X_t(:,:,i)))' * (vec(Y_t(:,:,i)-H_t(:,:,i)*X_t(:,:,i)));
                commut = vec(S * H_t(:,:,i) - H_t(:,:,i) * S);
                goal = goal + gamma * (commut' * commut);
            end
            minimize(goal)
            subject to 
                for z=1:T
                    H_t(:,:,z) == H_t(:,:,z)';
                end
        cvx_end

        cvx_begin quiet
            variable S(N, N) symmetric

            goal = beta * norm(vec(S),1) + alpha * beta * norm(vec(S),2);
            for i=1:T
                if single_S_observation
                    goal = goal + lambda_t(i) * norm(vec(S-Sn_t(:,:,1)),1);
                    goal = goal + alpha * lambda_t(i) * norm(vec(S-Sn_t(:,:,1)),2);
                else
                    goal = goal + lambda_t(i) * norm(vec(S-Sn_t(:,:,i)),1);
                    goal = goal + alpha * lambda_t(i) * norm(vec(S-Sn_t(:,:,i)),2);
                end
                commut = vec(S*H_t(:,:,i) - H_t(:,:,i)*S);
                goal = goal + gamma * (commut' * commut);
            end
            minimize(goal)
            subject to
                S >= 0;
                diag(S) == 0;
                S == S';
                sum(S, 2) == 1;
                
                if strcmp(stationnarity, 'Exact')
                    norm(Cy(:,:,1)*S-S*Cy(:,:,1),'fro') <= 0;
                end
                if strcmp(stationnarity, 'Estimated')
                    for a=1:T
                        norm(Cy_samp(:,:,a)*S-S*Cy_samp(:,:,a),'fro') <= delta;
                    end
                end        
        cvx_end
        
        if strcmp(cvx_status, 'Failed') || strcmp(cvx_status, 'Infeasible')
            disp(['WARNING: estHt_denS cvx status: ' cvx_status]);
            S = backupS;
            H_t = backupH_t;
        end
        
        backupS = S;
        backupH_t = H_t;
    end
end

