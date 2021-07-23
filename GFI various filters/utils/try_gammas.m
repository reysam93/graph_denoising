clear
rng(10)
addpath('./opt')

% Parameters
N = 20;
M = 20;
K = 4;
p = 0.25;
eps = .1;
P_N = [0 .01 .02 .03 .04 .05]; %[0 .02 .04 .06 .08 .1];
lambda = 1;
% Best gamma 0.1
gamma = 0.1;
max_iters = 10;
inc_gamma = false;
% delta = 0;
delta = 5e-3;
rec_th = 1e-3;
n_graphs = 10;

fmts = {'^-','v-','X-','o-','+-','*-'};
% models = {'H pinv','FI','RFI non st','RFI-R','RFI-D','RFI-Psi'};

models = {'gamma 1 true','gamma 1 false','gamma 0.1 true','gamma 0.1 false',...
    'gamma 0.01 true','gamma 0.01 false',};


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
    if delta == 0
        Cy = H*H';
    end
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
        
       if delta ~= 0
           Cy = Y*Y';
           Cy = Cy/norm(Cy,'fro');
       end
        
        disp(['      P_N ' num2str(p_n) ' (SNR: '...
            num2str(M*p_y/norm(Noise,'fro')^2) '): ' ])
        
        % Robust non st
        [H_rfi,A_rfi] = estH_non_st(X,Yn,A_pert,1,max_iters,true);
        if isempty(H_rfi)
            err_H(k,1,i) = 1;
            err_S(k,1,i) = 1;
        else
            err_H(k,1,i) = norm(H-H_rfi,'fro')^2/norm_H2;
            err_S(k,1,i) = norm(vec(A-A_rfi),1)/norm_A2;
        end
        
        [H_rfi,A_rfi] = estH_non_st(X,Yn,A_pert,1,max_iters,false);
        if isempty(H_rfi)
            err_H(k,2,i) = 1;
            err_S(k,2,i) = 1;
        else
            err_H(k,2,i) = norm(H-H_rfi,'fro')^2/norm_H2;
            err_S(k,2,i) = norm(vec(A-A_rfi),1)/norm_A2;
        end
        
        [H_rfi,A_rfi] = estH_non_st(X,Yn,A_pert,.1,max_iters,true);
        if isempty(H_rfi)
            err_H(k,3,i) = 1;
            err_S(k,3,i) = 1;
        else
            err_H(k,3,i) = norm(H-H_rfi,'fro')^2/norm_H2;
            err_S(k,3,i) = norm(vec(A-A_rfi),1)/norm_A2;
        end
        
        [H_rfi,A_rfi] = estH_non_st(X,Yn,A_pert,.1,max_iters,false);
        if isempty(H_rfi)
            err_H(k,4,i) = 1;
            err_S(k,4,i) = 1;
        else
            err_H(k,4,i) = norm(H-H_rfi,'fro')^2/norm_H2;
            err_S(k,4,i) = norm(vec(A-A_rfi),1)/norm_A2;
        end
        
        [H_rfi,A_rfi] = estH_non_st(X,Yn,A_pert,.01,max_iters,true);
        if isempty(H_rfi)
            err_H(k,5,i) = 1;
            err_S(k,5,i) = 1;
        else
            err_H(k,5,i) = norm(H-H_rfi,'fro')^2/norm_H2;
            err_S(k,5,i) = norm(vec(A-A_rfi),1)/norm_A2;
        end
        
        [H_rfi,A_rfi] = estH_non_st(X,Yn,A_pert,.01,max_iters,false);
        if isempty(H_rfi)
            err_H(k,6,i) = 1;
            err_S(k,6,i) = 1;
        else
            err_H(k,6,i) = norm(H-H_rfi,'fro')^2/norm_H2;
            err_S(k,6,i) = norm(vec(A-A_rfi),1)/norm_A2;
        end
        
        disp(['         Err H: ' num2str(err_H(k,:,i))])
    end
end
time = toc/60;
disp(['--- Ellapsed time: ' num2str(time) 'minutes ---'])

%% Plot
median_err_H = median(err_H,3);
median_err_S = median(err_S,3);

% figure()
% for i=1:length(models)
%     semilogy(P_N,median_err_H(:,i),fmts{i},'LineWidth',2,'MarkerSize',12)
%     hold on
% end
% hold off
% grid on
% xlabel('Normalized noise power')
% ylabel('Median error of H')
% title(['M: ' num2str(M) ' Eps: ' num2str(eps)])
% set(gca,'FontSize',20);
% legend(models,'FontSize',12)
% set(gcf, 'PaperPositionMode', 'auto')

figure()
for i=1:length(models)
    plot(P_N,median_err_H(:,i),fmts{i},'LineWidth',2,'MarkerSize',12)
    hold on
end
hold off
grid on
xlabel('Normalized noise power')
ylabel('Median error of H')
title(['M: ' num2str(M) ' Eps: ' num2str(eps) 'Gamma: ' num2str(gamma)...
    '- inc: ' num2str(inc_gamma)])
set(gca,'FontSize',20);
legend(models,'FontSize',12)
set(gcf, 'PaperPositionMode', 'auto')

figure()
for i=1:length(models)
    semilogy(P_N,median_err_S(:,i))
    hold on
end
hold off
legend(models)
xlabel('Normalized noise power')
ylabel('Median error of S')
title(['M: ' num2str(M) ' Eps: ' num2str(eps)])
grid on