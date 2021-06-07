% Generate samples from a dirichlet distribution.
% Input:
%   a: 1 by K dimensional vector
%   n: number of samples
% Output:
%   p: n by K dimensional probability vector
function p = randdir(a,n)

K = length(a);

if size(a,2)>size(a,1)
   a = a';
end;   

if 0   
    for i = 1:n
        p(:,i) = gamrnd(a,1);
		p(:,i) = theta(:,i)/sum(theta(:,i));
        p = p'
    end;       
else
    
    % faster version
    p = gamrnd(repmat(a,1,n),1,K,n);   
    p = p./repmat(sum(p,1),K,1);
    p = p';
end;

return;