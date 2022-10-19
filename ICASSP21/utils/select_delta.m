clear
rng(10)
addpath('./opt')

% Parameters
N = 20;
MM_cov = [100 250 500 1000 5000];
Max_M = 200;
L = 4;
p = 0.25;
eps = 0.1;
lambda = 1;
Deltas = [1e-4 2.5e-4 5e-4 7.5e-4 1e-3 2.5e-3 5e-3 7.5e-3 1e-2 2.5e-2 5e-2...
    7.5e-2 1e-1];  %9e-3;
p_n = 0.1;

models = {'Reg','Den','Den-Psi'};
n_graphs = 25;
err_h = zeros(length(models),length(Deltas),length(MM_cov),n_graphs);
err_H = zeros(length(models),length(Deltas),length(MM_cov),n_graphs);
err_S = zeros(length(models),length(Deltas),length(MM_cov),n_graphs);
tic
for i=1:n_graphs
    % Create A and get eigendecomposition
    A = generate_connected_ER(N, p);
    [V, Lambda] = eig(A);
    Psi = fliplr(vander(diag(Lambda)));
    Psi = Psi(:,1:L);
    
    % Perturbate A
    W = triu(rand(N)<eps,1);
    W = W+W';
    A_pert = double(xor(A,W));
    [V_pert, Lambda_pert] = eig(A_pert);
    Psi_pert = fliplr(vander(diag(Lambda_pert)));
    pert_links = norm(A-A_pert,'fro')^2/(N*(N-1));
    
    % Create filter (independent of M)
    h = 2*rand(L,1)-1;
    h = h/norm(h);
    H = V*diag(Psi(:,1:L)*h)*V';
    
    disp(['Graph ' num2str(i) ' (' num2str(pert_links) ' perturbed links): ' ])
    for j=1:length(MM_cov)
        M_cov = MM_cov(j);
        M = min(Max_M, M_cov);
        
        % Create data 
        X = randn(N,M_cov);
        Y = H*X;
        p_y = norm(Y,'fro')^2/M_cov;
        X = X(:,1:M);
        Yn = Y + randn(N,M_cov)*sqrt(p_y*p_n/N);
        
        % Compute original and sample covarianze with unit norm
        Cy = H*H';
        Cy = Cy/norm(Cy,'fro');
        Cy_samp = Yn*Yn';
        Cy_samp = Cy_samp/norm(Cy_samp,'fro');
        disp(['   M_cov: ' num2str(M_cov)])
        
        Yn = Yn(:,1:M);
        
        for k=1:length(Deltas)
            delta = Deltas(k);
            
            % Regularization model (based on Cy)
            [H_reg,A_reg] = estH_regS_2(X,Yn,A_pert,Cy_samp,lambda,delta);
            if isempty(H_reg)
                err_h(1,k,j,i) = 1;
                err_H(1,k,j,i) = 1;
                err_S(1,k,j,i) = 1;
            else
                [V_reg, Lambda_reg] = eig(A_reg);
                Psi_den = fliplr(vander(diag(Lambda_reg)));
                h_reg = pinv(Psi_den(:,1:L))*diag(V_reg'*H_reg*V_reg);
                err_h(1,k,j,i) = norm(h-h_reg)^2;
                err_S(1,k,j,i) = norm(A-A_reg,'fro')^2/norm(A,'fro')^2;
                err_H(1,k,j,i) = norm(H-H_reg,'fro')^2/norm(H,'fro')^2;
            end
            
            % Denoising model
            [H_den,A_den] = estH_denS_2(X,Yn,A_pert,Cy_samp,delta);
            if isempty(H_den)
                err_h(2,k,j,i) = 1;
                err_H(2,k,j,i) = 1;
                err_S(2,k,j,i) = 1;
            else
                [V_den, Lambda_den] = eig(A_den);
                Psi_den = fliplr(vander(diag(Lambda_den)));
                h_den = pinv(Psi_den(:,1:L))*diag(V_den'*H_den*V_den);
                err_h(2,k,j,i) = norm(h-h_den)^2;
                err_S(2,k,j,i) = norm(A-A_den,'fro')^2/norm(A,'fro')^2;
                err_H(2,k,j,i) = norm(H-H_den,'fro')^2/norm(H,'fro')^2;
            end
            
            % Denoising model using Psi
            [h_den_psi,A_den_psi] = estH_denS_Psi_2(X,Yn,A_pert,Cy_samp,L,delta);
            [V_den_psi, Lambda_den_psi] = eig(A_den_psi);
            Psi_den_psi = fliplr(vander(diag(Lambda_den_psi)));
            H_den_Psi = V_den_psi*diag(Psi_den_psi(:,1:L)*h_den_psi)*V_den_psi';
            if isempty(h_den_psi)
                err_h(3,k,j,i) = 1;
                err_H(3,k,j,i) = 1;
                err_S(3,k,j,i) = 1;
            else
                err_h(3,k,j,i) = norm(h-h_den_psi)^2;
                err_H(3,k,j,i) = norm(H-H_den_Psi,'fro')^2/norm(H,'fro')^2;
                err_S(3,k,j,i) = norm(A-A_den_psi,'fro')^2/norm(A,'fro')^2;
            end
            disp(['      Delta ' num2str(delta) ': Err h: ' num2str(err_h(:,k,j,i)')])
        end
    end
end
time = toc/60;
disp(['--- Ellapsed time: ' num2str(time) 'minutes ---'])

%% Plot
median_err_h = median(err_h,4);
% median_log_err_h = median(log10(err_h),4);
median_err_H = median(err_H,4);
% median_log_err_H = median(log10(err_H),4);
median_err_S = median(err_S,4);

for i=1:length(models)
    model = models{i};
    
%     figure()
%     imagesc(squeeze(median_err_h(i,:,:)))
%     ylabel('Deltas')
%     xlabel('Samples Cy')
%     yticklabels(Deltas)
%     xticklabels(MM_cov)
%     xticks(1:length(MM_cov))
%     yticks(1:length(Deltas))
%     title(['Err (h) - model ' model ' Pn-' num2str(p_n) ' M-' num2str(Max_M)])
%     colorbar()
    
%     figure()
%     imagesc(squeeze(median_log_err_h(i,:,:)))
%     ylabel('Deltas')
%     xlabel('Samples Cy')
%     yticklabels(Deltas)
%     xticklabels(MM_cov)
%     xticks(1:length(MM_cov))
%     title(['Log Err (h) - model ' model ' Pn: ' num2str(p_n)])
%     colorbar()
    
    figure()
    imagesc(squeeze(median_err_H(i,:,:)))
    ylabel('Deltas')
    xlabel('Samples Cy')
    yticklabels(Deltas)
    xticklabels(MM_cov)
    xticks(1:length(MM_cov))
    yticks(1:length(Deltas))
    title(['Err (H) - model ' model ' Pn-' num2str(p_n) ' M-' num2str(Max_M)])
    colorbar()
    
%     figure()
%     imagesc(squeeze(median_log_err_H(i,:,:)))
%     ylabel('Deltas')
%     xlabel('Samples Cy')
%     yticklabels(Deltas)
%     xticklabels(MM_cov)
%     xticks(1:length(MM_cov))
%     title(['Log Err (H) - model ' model ' Pn: ' num2str(p_n)])
%     colorbar()

%     figure()
%     imagesc(MM_cov,Deltas,squeeze(median_err_S(i,:,:)))
%     ylabel('Deltas')
%     xlabel('Samples Cy')
% %     yticklabels(Deltas)
% %     xticklabels(MM_cov)
% %     xticks(1:length(MM_cov))
% %     yticks(1:length(Deltas))
%     title(['Err (S) - model ' model ' Pn-' num2str(p_n) ' M-' num2str(Max_M)])
%     colorbar()
end