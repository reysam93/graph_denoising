clear
rng(10)
addpath('./opt')
addpath('./utils')

% Parameters
N = 20;
M = 10;
K = 4;
p = 0.25;
eps = .1;
n_graphs = 10;100;
P_N = [0];[0 .01 .02 .03 .04 .05];

lambda = 1;
gamma = 0.1;
max_iters = 20;
inc_gamma = true;

delta = 3e-5;
delta_st = 1e-4;
delta_st_samp = 5e-3;
rec_th = 1e-3;

fmts = {'^-','v-','X-','o-'};
models = {'FI','RFI iter','RFI-D Cy','RFI-D Cy est'};

err_H = zeros(length(P_N),length(models),n_graphs);
err_S = zeros(length(P_N),length(models),n_graphs);
tic
for i=1:n_graphs
    disp(['Graph ' num2str(i) ':'])
    A = generate_connected_ER(N, p);
    norm_A2 = N*(N-1);
    [V, Lambda] = eig(A);
    Psi = fliplr(vander(diag(Lambda)));
    
    % Create data
    X = randn(N,M)/sqrt(N);
    h = 2*rand(K,1)-1;
    h = h/norm(h);
    H = V*diag(Psi(:,1:K)*h)*V';
    norm_H2 = norm(H,'fro')^2;
    Y = H*X;
    Cy = H*H';
    p_y = norm(Y,'fro')^2/M;
    
    % Perturbate graph
    W = triu(rand(N)<eps,1);
    W = W+W';
    A_pert = double(xor(A,W));
    [V_pert, Lambda_pert] = eig(A_pert);
    Psi_pert = fliplr(vander(diag(Lambda_pert)));
    pert_links = norm(A-A_pert,'fro')^2/(N*(N-1));
    disp(['   Eps ' num2str(eps) ' (' num2str(pert_links) ' pert links): ' ])
    for k=1:length(P_N)
        p_n = P_N(k);
        Noise = randn(N,M)*sqrt(p_y*p_n/N);
        Yn = Y + Noise;
        
        Cy_samp = Y*Y';
        Cy_samp = Cy_samp/norm(Cy_samp,'fro');
        
        disp(['      P_N ' num2str(p_n) ' (SNR: '...
            num2str(M*p_y/norm(Noise,'fro')^2) '): ' ])
        
%         % Unperturbed model
%         H_unp = estH_unpertS(X,Yn,A_pert);
%         if isempty(H_unp)
%             err_H(k,1,i) = 1;
%             err_S(k,1,i) = 1;
%         else
%             err_H(k,1,i) = norm(H-H_unp,'fro')^2/norm_H2;
%             err_S(k,1,i) = norm(vec(A-A_pert),1)/norm_A2;
%         end
        
        % Robust non st
        [H_rfi,A_rfi] = estH_non_st(X,Yn,A_pert,lambda,gamma,max_iters,inc_gamma);
        if isempty(H_rfi)
            err_H(k,2,i) = 1;
            err_S(k,2,i) = 1;
        else
            err_H(k,2,i) = norm(H-H_rfi,'fro')^2/norm_H2;
            err_S(k,2,i) = norm(vec(A-A_rfi),1)/norm_A2;
        end
        
%         % Denoising model - perfect Cy
%         [H_den,A_den] = estH_denS(X,Yn,A_pert,Cy,0);
%         if isempty(H_den)
%             err_H(k,3,i) = 1;
%             err_S(k,3,i) = 1;
%         else
%             err_H(k,3,i) = norm(H-H_den,'fro')^2/norm_H2;
%             err_S(k,3,i) = norm(vec(A-A_den),1)/norm_A2;
%         end
        
        % Denoising model - sampled Cy
        [H_den,A_den] = estH_denS(X,Yn,A_pert,Cy_samp,delta);
        if isempty(H_den)
            err_H(k,4,i) = 1;
            err_S(k,4,i) = 1;
        else
            err_H(k,4,i) = norm(H-H_den,'fro')^2/norm_H2;
            err_S(k,4,i) = norm(vec(A-A_den),1)/norm_A2;
        end

        disp(['         Err H: ' num2str(err_H(k,:,i))])
    end
end
time = toc/60;
disp(['--- Ellapsed time: ' num2str(time) 'minutes ---'])

%% Plot
median_err_H = median(err_H,3);

figure()
for i=1:length(models)
    plot(P_N,median_err_H(:,i),fmts{i},'LineWidth',2,'MarkerSize',12)
    hold on
end
hold off
grid on
xlabel('Normalized noise power')
ylabel('Median error')
set(gca,'FontSize',20);
legend(models,'FontSize',14,'interpreter','latex')
set(gcf, 'PaperPositionMode', 'auto')
