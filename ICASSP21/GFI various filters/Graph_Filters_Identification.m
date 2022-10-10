%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GRAPH FILTERS IDENTIFICATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

% Set a random seed and define directory path
rng(1)
addpath('./opt')
addpath('./utils')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters

% Data
N = 15;                   % Graph size
T = 5;                    % Number of filters to jointly identitfy
MM_cov = [5, 10, 12, 15]; % Number of input/output pairs for each filter     
Noise_Model = false;      % Add noise to the perturbed graphs
LL = [4];                  
p_n = 0.1;                
p_A = 0.1;

% Optimization
eps = 0.1;                % Edge perturbation probability
gamma = 1;                % Commutation term weight (see (4))
lambda_t = ones(T,1);  % Observed graph terms weights
n_signals = 1;            % Nb of experiments to be run for each settings
delta = 1;                % Stationnarity tolerance
 
% Outputs
models = {
    'Least Squares',
    '1 iteration',
    '10 iterations',
    '10 iterations exact Cy',
    '10 iterations estimated Cy',
    '10 iterations single GSO obs.'};

% Type of optimization
func = {@estHt_Least_Squares, @estHt_denS, @estHt_denS, @estHt_denS, @estHt_denS, @estHt_denS};

% T, max_iters, lambda_t, gamma, beta, alpha, gamma_inc,
% single_S_observation, Stationnarity type (= 'No', 'Exact', 'Estimated'),
% delta
parameters = {
    {T},
    {T, 1, lambda_t, gamma, 1, 0, 1, false, 'No', delta},
    {T, 10, lambda_t, gamma, 1, 0, 1, false, 'No', delta},
    {T, 10, lambda_t, gamma, 1, 0, 1, false, 'Exact', delta},
    {T, 10, lambda_t, gamma, 1, 0, 1, false, 'Estimated', delta},
    {T, 10, lambda_t, gamma, 1, 0, 1, true, 'No', delta}};
    
markers = {'v','o','+','v','o', 'v'};
lines = {'-','--'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

% Get adjacency matrix

% load('./datasets/karate.mat')
% A = adjacency(graph(edges(:,1),edges(:,2)));

A = round(rand(N,N));
A = tril(A, -1) + triu(A', 1);

norm_A2 = N*(N-1);
[V, Lambda] = eigs(A);
Psi = fliplr(vander(diag(Lambda)));

% Errors
err_H = zeros(length(models),length(MM_cov),length(LL),n_signals);
err_S = zeros(length(models),length(MM_cov),length(LL),n_signals);
err_S_fro = zeros(length(models),length(MM_cov),length(LL),n_signals);
C_err = zeros(length(MM_cov),length(LL),n_signals);

S_t = zeros(N, N, T);
H_t = zeros(N, N, T);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:n_signals    
    % Perturbate A, T times
    for a=1:T
        W = triu(rand(N)<eps,1);
        W = W+W';
        
        A_pert = double(xor(A,W));
        [V_pert, Lambda_pert] = eig(A_pert);
        Psi_pert = fliplr(vander(diag(Lambda_pert)));
        pert_links = norm(A-A_pert,'fro')^2/(N*(N-1));
        
        % Noise (only for noise model of perturbation)
        if Noise_Model
            A_pert = A_pert + randn(N,N)*sqrt(p_A * norm(vec(A),'fro')^2 / N^2);
        end
        
        S_t(:,:,a) = A_pert;
    end

    for k=1:length(LL)
        L=LL(k);
        
        for a=1:T
            % Create filter (independent of M)
            h = 2*rand(L,1)-1;
            h = h/norm(h);
            H = V*diag(Psi(:,1:L)*h)*V';

            disp(['   L: ' num2str(L)])

            H_t(:,:,a) = H;
        end
        
        norm_H2 = norm(vec(H_t),'fro')^2;

        for j=1:length(MM_cov)

            M_cov = MM_cov(j);
            
            % Create data
            
            Y_t = zeros(N,M_cov,T);
            X_t = randn(N,M_cov,T);
            
            for b=1:T
                Y_t(:,:,b) = H_t(:,:,b)*X_t(:,:,b);
            end

            p_y = norm(vec(Y_t),'fro')^2/M_cov;
            Noise = randn(N,M_cov,T)*sqrt(p_y*p_n/N);
            Yn = Y_t + Noise;
            
            % True covariance Cy
            Cy_t = zeros(N, N, T);
            
            % Estimated covariances Cy_samp_t
            Cy_samp_t = zeros(N, N, T);

            for a=1:T
                Cy = H_t(:,:,a)*H_t(:,:,a)';
                Cy = Cy/norm(Cy,'fro');
                Cy_samp = Yn(:,:,a)*Yn(:,:,a)';
                Cy_samp = Cy_samp/norm(Cy_samp,'fro');
                Cy_t(:,:,a) = Cy;
                Cy_samp_t(:,:,a) = Cy_samp;
            end

            % Optimize and save errors
            for m=1:length(models)
                disp(['Model ' models(m)]);
                
                [H_res, S_res] = func{m}(X_t, Y_t, S_t, Cy_t, Cy_samp_t, parameters{m});
                
                err_H(m,j,k,i) = norm(vec(H_t)-vec(H_res),'fro')^2/norm_H2;
                err_S(m,j,k,i) = norm(vec(A-S_res),1)/norm_A2;
                err_S_fro(m,j,k,i) = norm(A-S_res,'fro')^2/norm_A2;
            end
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rec_th = 1e-3;


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
