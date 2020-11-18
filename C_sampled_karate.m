clear
rng(10)
addpath('./opt')
addpath('./utils')

% Parameters
MM_cov = [50 100 250 500 1000 5000];
Max_M = 200;
LL = [4];
eps = 0.1;
lambda = 1;
delta = 5e-3;
gamma = 1;
inc_gamma = true;

max_iters = 20;
p_n = 0.1;
rec_th = 1e-3;

models = {'FI','RFI-R','RFI-D'};
markers = {'v','o','+','X'};
lines = {'-','--'};

n_signals = 100;

% Get adjacency matrix
load('./datasets/karate.mat')
A = adjacency(graph(edges(:,1),edges(:,2)));
N = size(A,1);
norm_A2 = N*(N-1);
[V, Lambda] = eigs(A);
Psi = fliplr(vander(diag(Lambda)));

err_H = zeros(length(models),length(MM_cov),length(LL),n_signals);
err_S = zeros(length(models),length(MM_cov),length(LL),n_signals);
err_S_fro = zeros(length(models),length(MM_cov),length(LL),n_signals);
C_err = zeros(length(MM_cov),length(LL),n_signals);
tic
for i=1:n_signals    
    % Perturbate A
    W = triu(rand(N)<eps,1);
    W = W+W';
    A_pert = double(xor(A,W));
    [V_pert, Lambda_pert] = eig(A_pert);
    Psi_pert = fliplr(vander(diag(Lambda_pert)));
    pert_links = norm(A-A_pert,'fro')^2/(N*(N-1));
    
    disp(['Signal ' num2str(i) ' (' num2str(pert_links) ' perturbed links): ' ])
    
    for k=1:length(LL)
        L=LL(k);

        % Create filter (independent of M)
        h = 2*rand(L,1)-1;
        h = h/norm(h);
        H = V*diag(Psi(:,1:L)*h)*V';
        norm_H2 = norm(H,'fro')^2;
        disp(['   L: ' num2str(L)])
        for j=1:length(MM_cov)
            M_cov = MM_cov(j);
            M = min(Max_M, M_cov);
            
            % Create data
            X = randn(N,M_cov);
            Y = H*X;
            p_y = norm(Y,'fro')^2/M_cov;
            Noise = randn(N,M_cov)*sqrt(p_y*p_n/N);
            Yn = Y + Noise;
            
            % Compute original and sample covarianze with unit norm
            Cy = H*H';
            Cy = Cy/norm(Cy,'fro');
            Cy_samp = Yn*Yn';
            Cy_samp = Cy_samp/norm(Cy_samp,'fro');
            C_err(j,k,i) = norm(Cy_samp-Cy,'fro');
            disp(['      M_cov: ' num2str(M_cov) ' (err w.r.t. original Cy: '...
                num2str(C_err(j,i)) ')' ])
            
            X = X(:,1:M);
            Yn = Yn(:,1:M);
            
            % Unperturbed model
            H_unp = estH_unpertS(X,Yn,A_pert);
            if isempty(H_unp)
                error('H_unp EMPTY!')
            end
            err_H(1,j,k,i) = norm(H-H_unp,'fro')^2/norm_H2;
            err_S(1,j,k,i) = norm(vec(A-A_pert),1)/norm_A2;
            err_S_fro(1,j,k,i) = norm(A-A_pert,'fro')^2/norm_A2;
            
            % Regularization model (based on Cy)
            [H_reg,A_reg] = estH_regS(X,Yn,A_pert,Cy_samp,lambda,delta);
            if isempty(H_reg)
                err_H(3,j,k,i) = 1;
                err_S(3,j,k,i) = 1;
                err_S_fro(3,j,k,i) = 1;
            else
                err_H(3,j,k,i) = norm(H-H_reg,'fro')^2/norm_H2;
                err_S(3,j,k,i) = norm(vec(A-A_reg),1)/norm_A2;
                err_S_fro(3,j,k,i) = norm(A-A_reg,'fro')^2/norm_A2;
            end
            
            % Denoising model
            [H_den,A_den] = estH_denS(X,Yn,A_pert,Cy_samp,delta);
            if isempty(H_den)
                err_H(4,j,k,i) = 1;
                err_S(4,j,k,i) = 1;
                err_S_fro(4,j,k,i) = 1;
            else
                err_H(4,j,k,i) = norm(H-H_den,'fro')^2/norm_H2;
                err_S(4,j,k,i) = norm(vec(A-A_den),1)/norm_A2;
                err_S_fro(4,j,k,i) = norm(A-A_den,'fro')^2/norm_A2;
            end
            
            disp(['         Err H: ' num2str(err_H(:,j,k,i)')])
            disp(['         Err S: ' num2str(err_S(:,j,k,i)')])
        end
    end
end
time = toc/60;
disp(['--- Ellapsed time: ' num2str(time) 'minutes ---'])

%% Display results
med_C_err = median(C_err,3);
for j=1:length(LL)
    disp(['Median C err (L=' num2str(LL(j)) '): ' num2str(med_C_err(:,j)')])
end

med_err_H = median(err_H,4);
med_err_S = median(err_S_fro,4);
rec_H = sum(err_H<rec_th*10,4)/n_signals;

% Plot err H
leg = cell(length(LL)*length(models),1);
figure()
for j=1:length(LL)
    for i=1:length(models)
        leg{(j-1)*length(models)+i}=[models{i} ', L=' num2str(LL(j))];
        fmt = [markers{i} lines{j}];
        loglog(MM_cov,med_err_H(i,:,j),fmt,'LineWidth',2,'MarkerSize',12)
        hold on
    end
end
grid on
axis tight
xlabel('Number of samples')
xticks(MM_cov)
xticklabels(MM_cov)
ylabel('(b) Filter median error')
ylim([0.001 1])
set(gca,'FontSize',20);
legend(models,'FontSize',12,'location','southwest')
set(gcf, 'PaperPositionMode', 'auto')

% Plot err S
leg = cell(length(LL)*length(models),1);
figure()
for j=1:length(LL)
    for i=1:length(models)
        leg{(j-1)*length(models)+i}=[models{i} ' - L: ' num2str(LL(j))];
        fmt = [markers{i} lines{j}];
        semilogx(MM_cov,med_err_S(i,:,j),fmt,'LineWidth',2,'MarkerSize',12)
        hold on
    end
end
grid on
axis tight
xlabel('Number of samples')
xticks(MM_cov)
ylabel('(a) Graph median error')
set(gca,'FontSize',20);
legend(models,'FontSize',12,'location','southwest')
set(gcf, 'PaperPositionMode', 'auto')
