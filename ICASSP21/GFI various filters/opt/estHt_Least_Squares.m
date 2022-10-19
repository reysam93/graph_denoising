function [H_t, S] = estH_time_varying(X_t,Y_t, Sn_t, Cy, Cy_samp, param)
%ESTH_ Summary of this function goes here
%   Detailed explanation goes here

    T = param{1, 1};
    gamma = 0;
    
    s = size(X_t);
    N = s(1);  
        
    disp(['LEAST SQUARES '])
    S = Sn_t(:,:,1);
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
    
    S = zeros(N,N);
    for i=1:T
        S = S + Sn_t(:,:,i);
    end
    S = S/T;
    
end

