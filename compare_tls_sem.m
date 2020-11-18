clear
rng(10)
addpath('./utils')

% Parameters
N = 20;
M_cov = 200;
M = min(200,M_cov);
L = 5;
p = 0.25;
EPS = [0, .1, .2, .3, .4, .5];
p_n = 0.1;

% Regs
lambda = 1;
delta = 5e-3;

% Regs for TLS-SEM - Computed for M=200
lambda_sem1_sem = 0.5;
lambda_sem2_sem = 0.75;
lambda_sem1_poly = 0.1;
lambda_sem2_poly = 0.01;

max_iters = 10;

n_graphs = 100;
models = {'TLS-SEM, SEM','RFI-R, SEM','RFI-D, SEM',...
    'TLS-SEM, H','RFI-R, H','RFI-D, H'};
fmts = {'^-','o-','+-','^--','o--','+--'};

err_H = zeros(length(models),length(EPS),n_graphs);
err_S = zeros(length(models),length(EPS),n_graphs);
err_Y = zeros(length(models),length(EPS),n_graphs);
tic

for i=1:n_graphs
    disp(['Graph ' num2str(i) ':'])
    A = generate_connected_ER(N, p);
    [V, Lambda] = eig(A);
    Psi = fliplr(vander(diag(Lambda)));
    Psi = Psi(:,1:L);
    norm_A2 = norm(A,'fro')^2;
    
    % Create data 
    X = randn(N,M_cov)/sqrt(N);
    
    % SEM generator model
    H_sem = pinv(eye(N)-A);
    norm_H2_sem = norm(H_sem,'fro')^2;
    Y_sem = H_sem*X;
    Yn_sem = Y_sem + randn(N,M_cov)*sqrt(norm(Y_sem,'fro')^2*p_n/(N*M_cov));
    Cy_samp_sem = Yn_sem*Yn_sem';
    Cy_samp_sem = Cy_samp_sem/norm(Cy_samp_sem,'fro');
    
    % Polynomial generator model
    h = 2*rand(L,1)-1;
    h = h/norm(h);
    H_poly = V*diag(Psi*h)*V';
    norm_H2_poly = norm(H_poly,'fro')^2;
    Y_poly = H_poly*X;
    Yn_poly = Y_poly + randn(N,M_cov)*sqrt(norm(Y_poly,'fro')^2*p_n/(N*M_cov));
    Cy_samp_poly = Yn_poly*Yn_poly';
    Cy_samp_poly = Cy_samp_poly/norm(Cy_samp_poly,'fro');
    
    % Limit the number of samples for denoising models
    Xm = X(:,1:M);
    Yn_m_sem = Yn_sem(:,1:M);
    Yn_m_poly = Yn_poly(:,1:M);

    for j=1:length(EPS)
        eps = EPS(j);
        
        % Perturbate graph
        W = triu(rand(N)<eps,1);
        W = W+W';
        A_pert = double(xor(A,W));
        pert_links = norm(A-A_pert,'fro')^2/(N*(N-1));
        disp(['   Eps ' num2str(eps) ' (' num2str(pert_links) ' pert links): ' ])
        
        % TLS-SEM (noise aware) - SEM data
        [H_hat,A_hat] = estH_tls_sem_noise(X,Yn_sem,A_pert,lambda_sem1_sem,...
            lambda_sem2_sem,max_iters);
        if isempty(H_hat)
            err_H(1,j,i) = 1;
            err_S(1,j,i) = 1;
            err_Y(1,j,i) = 1;
        else
            err_H(1,j,i) = norm(H_sem-H_hat,'fro')^2/norm_H2_sem;
            err_S(1,j,i) = norm(A-A_hat,'fro')^2/norm_A2;
            err_Y(1,j,i) = norm(Y_sem-H_hat*X,'fro')^2/norm(Y_sem,'fro')^2;
        end

        % Reg model - SEM data
        [H_hat,A_hat] = estH_regS(Xm,Yn_m_sem,A_pert,Cy_samp_sem,lambda,delta);
        if isempty(H_hat)
            err_H(2,j,i) = 1;
            err_S(2,j,i) = 1;
            err_Y(2,j,i) = 1;
        else
            err_H(2,j,i) = norm(H_sem-H_hat,'fro')^2/norm_H2_sem;
            err_S(2,j,i) = norm(A-A_hat,'fro')^2/norm_A2;
            err_Y(2,j,i) = norm(Y_sem-H_hat*X,'fro')^2/norm(Y_sem,'fro')^2;
        end
        
        % Denoising model - SEM data
        [H_hat,A_hat] = estH_denS(Xm,Yn_m_sem,A_pert,Cy_samp_sem,delta);
        if isempty(H_hat)
            err_H(3,j,i) = 1;
            err_S(3,j,i) = 1;
            err_Y(3,j,i) = 1;
        else
            err_H(3,j,i) = norm(H_sem-H_hat,'fro')^2/norm_H2_sem;
            err_S(3,j,i) = norm(A-A_hat,'fro')^2/norm_A2;
            err_Y(3,j,i) = norm(Y_sem-H_hat*X,'fro')^2/norm(Y_sem,'fro')^2;
        end
        
        % TLS-SEM (noise aware) - Poly data
        [H_hat,A_hat] = estH_tls_sem_noise(X,Yn_poly,A_pert,lambda_sem1_poly,...
            lambda_sem2_poly,max_iters);
        if isempty(H_hat)
            err_H(4,j,i) = 1;
            err_S(4,j,i) = 1;
            err_Y(4,j,i) = 1;
        else
            err_H(4,j,i) = norm(H_poly-H_hat,'fro')^2/norm_H2_poly;
            err_S(4,j,i) = norm(A-A_hat,'fro')^2/norm_A2;
            err_Y(4,j,i) = norm(Y_poly-H_hat*X,'fro')^2/norm(Y_poly,'fro')^2;
        end

        % Reg model - SEM data
        [H_hat,A_hat] = estH_regS(Xm,Yn_m_poly,A_pert,Cy_samp_poly,lambda,delta);
        if isempty(H_hat)
            err_H(5,j,i) = 1;
            err_S(5,j,i) = 1;
            err_Y(5,j,i) = 1;
        else
            err_H(5,j,i) = norm(H_poly-H_hat,'fro')^2/norm_H2_poly;
            err_S(5,j,i) = norm(A-A_hat,'fro')^2/norm_A2;
            err_Y(5,j,i) = norm(Y_poly-H_hat*X,'fro')^2/norm(Y_poly,'fro')^2;
        end
        
        % Denoising model - SEM data
        [H_hat,A_hat] = estH_denS(Xm,Yn_m_poly,A_pert,Cy_samp_poly,delta);
        if isempty(H_hat)
            err_H(6,j,i) = 1;
            err_S(6,j,i) = 1;
            err_Y(6,j,i) = 1;
        else
            err_H(6,j,i) = norm(H_poly-H_hat,'fro')^2/norm_H2_poly;
            err_S(6,j,i) = norm(A-A_hat,'fro')^2/norm_A2;
            err_Y(6,j,i) = norm(Y_poly-H_hat*X,'fro')^2/norm(Y_poly,'fro')^2;
        end
        
        disp(['      Error H: ' num2str(err_H(:,j,i)')])
        disp(['      Error S: ' num2str(err_S(:,j,i)')])
    end
end
time = toc/60;
disp(['--- Ellapsed time: ' num2str(time) 'minutes ---'])
%% Plot
median_err_H = median(err_H,3);
median_err_S = median(err_S,3);
median_err_Y = median(err_Y,3);

figure()
for i=1:length(models)
    semilogy(EPS,median_err_H(i,:),fmts{i},'LineWidth',2,'MarkerSize',12)
    hold on
end
hold off
xlabel('Link pert. prob.')
ylabel('(c) Filter median error')
set(gca,'FontSize',20);
legend(models,'FontSize',12,'interpreter','latex')
grid on
set(gcf, 'PaperPositionMode', 'auto')

figure()
for i=1:length(models)
    semilogy(EPS,median_err_S(i,:),fmts{i})
    hold on
end
hold off
legend(models)
xlabel('Link pert. prob.')
ylabel('Median error of S')
title(['M: ' num2str(M_cov) ' Pn: ' num2str(p_n)])
grid on
