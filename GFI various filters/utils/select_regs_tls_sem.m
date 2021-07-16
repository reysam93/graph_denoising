clear
rng(10)
addpath('./opt')

% Parameters
N = 20;
M = 200;
L = 4;
p = 0.25;
eps = 0.1;
p_n = 0.1;
sem_model = false;

% For SEM model
if sem_model
    % For M=200
    Lambdas1 = [0.1 0.5 1 2 3];
    Lambdas2 = [0.25 0.5 0.75 1 1.25];
%     For M=1000
%     Lambdas1 = 1:5;
%     Lambdas2 = [0.25 0.5 0.75 1 1.25];
else
    % For M=200
    Lambdas1 = [0.05 0.1 0.5 1 1.5];
    Lambdas2 = [5e-4 1e-3 5e-3  1e-2 5e-2];
%     % For M=1000
%     Lambdas1 = 3:2:11;
%     Lambdas2 = [5e-4 1e-3 5e-3  1e-2 5e-2 .1];
end


max_iters = 10;

n_graphs = 25;
err_H = zeros(length(Lambdas1),length(Lambdas2),n_graphs);
err_S = zeros(length(Lambdas1),length(Lambdas2),n_graphs);
err_Y = zeros(length(Lambdas1),length(Lambdas2),n_graphs);
tic
for i=1:n_graphs
    % Create and perturb A
    A = generate_connected_ER(N, p);
    norm_A2 = norm(A,'fro')^2;
    W = triu(rand(N)<eps,1);
    W = W+W';
    A_pert = double(xor(A,W));

    % Create data
    X = randn(N,M)/sqrt(N);
    
    if sem_model
        H =  pinv(eye(N)-A);
    else
        [V, Lambda] = eig(A);
        Psi = fliplr(vander(diag(Lambda)));
        Psi = Psi(:,1:L);
        h = 2*rand(L,1)-1;
        h = h/norm(h);
        H = V*diag(Psi*h)*V';
    end
    
    norm_H2 = norm(H,'fro')^2;
    Y = H*X;
    Yn = Y + randn(N,M)*sqrt(norm(Y,'fro')^2*p_n/(N*M));
    
    disp(['Graph ' num2str(i) ':'])
    for j=1:length(Lambdas2)
        lambda2 = Lambdas2(j);
        disp(['   lambda2: ' num2str(lambda2)])
        for k=1:length(Lambdas1)
            lambda1 = Lambdas1(k);
            
            [H_hat,A_hat] = estH_tls_sem_noise(X,Yn,A_pert,lambda1,...
                lambda2,max_iters);
            if isempty(H_hat)
                err_H(k,j,i) = 1;
                err_S(k,j,i) = 1;
                err_Y(k,j,i) = 1;
            else
                err_H(k,j,i) = norm(H-H_hat)^2/norm_H2;
                err_S(k,j,i) = norm(A-A_hat,'fro')^2/norm_A2;
                err_Y(k,j,i) = norm(Yn-H_hat*X,'fro')^2/norm(Yn,'fro')^2;
            end
            
            
            disp(['      lambda1 ' num2str(lambda1) ': Err H: '...
                num2str(err_H(k,j,i)) ' Err S: ' num2str(err_S(k,j,i))])
        end
    end
end
time = toc/60;
disp(['--- Ellapsed time: ' num2str(time) 'minutes ---'])

%% Plot
median_err_H = median(err_H,3);
median_err_S = median(err_S,3);
median_err_Y = median(err_Y,3);    

figure()
imagesc(median_err_H)
xlabel('Lambdas2')
ylabel('Lambdas1')
xticklabels(Lambdas2)
yticklabels(Lambdas1)
xticks(1:length(Lambdas2))
yticks(1:length(Lambdas1))
title(['SEM data: ' num2str(sem_model) ' Err (H): Pn-' num2str(p_n)...
    ' M-' num2str(M)])
colorbar()

figure()
imagesc(median_err_S)
xlabel('Lambdas2')
ylabel('Lambdas1')
xticklabels(Lambdas2)
yticklabels(Lambdas1)
xticks(1:length(Lambdas2))
yticks(1:length(Lambdas1))
title(['SEM data: ' num2str(sem_model) ' Err (S): Pn-' num2str(p_n)...
    ' M-' num2str(M)])
colorbar()

figure()
imagesc(median_err_Y)
xlabel('Lambdas2')
ylabel('Lambdas1')
xticklabels(Lambdas2)
yticklabels(Lambdas1)
xticks(1:length(Lambdas2))
yticks(1:length(Lambdas1))
title(['SEM data: ' num2str(sem_model) ' Err (Y): Pn-' num2str(p_n)...
    ' M-' num2str(M)])
colorbar()
