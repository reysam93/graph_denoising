function [H] = estH_unpertS(X,Y,S)
    [V,~] = eig(S);
    Z = krb(X'*V,V);
    h_freq = Z\vec(Y);
    H = V*diag(h_freq)*V';
end