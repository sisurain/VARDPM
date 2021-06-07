function [z, Theta, w, llh] = oneDPM(X, h, alpha, theta)
% One time (for initialization) collapsed Gibbs sampling for Dirichlet process (infinite) mixture model. 
% Input: 
%   X: d x n data matrix
%   h: 1 x n in e^(h/2) of SV
%   alpha: concentration parameter for Dirichlet process prior
%   theta: class object for prior of component distribution (such as Gauss)
% Output:
%   z: 1 x n cluster label
%   Theta: 1 x k structure of trained components
%   w: 1 x k component weight vector
%   llh: loglikelihood
T = size(X,2);
Theta = {};
nk = [];
z = zeros(1,T);
llh = 0;
for i = randperm(T)
    x = X(:,i);
    ht = h(i);
    Pk = log(nk)+cellfun(@(t) t.logPredPdf(x,ht), Theta);
    P0 = log(alpha)+theta.logPredPdf(x,ht);
    p = [Pk,P0];
    llh = llh+sum(p-log(T));
    
    p = p - max(p,[],2);
    p = exp(p);
    p = p./sum(p,2);
    
    k = randmn(p);
    if k == numel(Theta)+1
        Theta{k} = theta.clone().addSample(x,ht);
        nk = [nk,1];
    else
        Theta{k} = Theta{k}.addSample(x,ht);
        nk(k) = nk(k)+1;
    end
    z(i) = k;
end
w = nk/T;