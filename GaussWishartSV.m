% Class for Gaussian-Wishart distribution used by Dirichlet process with SV
classdef GaussWishartSV
        properties
         kappa_
         mu_
         nu_
         U_ %instead of S, the precision matrix
     end
     
     methods
         function obj = GaussWishartSV(kappa,mu,nu,S)
             U = chol(S+kappa*(mu*mu'));
             obj.kappa_ = kappa;
             obj.mu_ = mu;
             obj.nu_ = nu;
             obj.U_ = U;
         end
         
         function obj = clone(obj)
         end
         
         function obj = addData(obj, X, h) % X is d by n, h is 1 by n
             kappa0 = obj.kappa_;
             mu0 = obj.mu_;
             nu0 = obj.nu_;
             U0 = obj.U_;
             
             exph = exp(-h/2);
             exph2 = exp(-h);
             n = size(X,2);
             kappa = kappa0+sum(exph2);
             mu = (kappa0*mu0+sum(exph2.*X,2))/kappa;
             nu = nu0+n;
             U = chol(U0'*U0+(exph.*X)*(exph.*X)');
             obj.kappa_ = kappa;
             obj.mu_ = mu;
             obj.nu_ = nu;
             obj.U_ = U;
         end
        
         function obj = addSample(obj, x, h) %x is d by 1
             kappa = obj.kappa_;
             mu = obj.mu_;
             nu = obj.nu_;
             U = obj.U_;
             
             exph = exp(-h/2);
             exph2 = exp(-h);
             kappa = kappa+exph2;
             mu = mu+exph2*(x-mu)/kappa;
             nu = nu+1;
             U = cholupdate(U,exph*x,'+');
             
             obj.kappa_ = kappa;
             obj.mu_ = mu;
             obj.nu_ = nu;
             obj.U_ = U;
         end
         
         function obj = delSample(obj, x, h)
             kappa = obj.kappa_;
             mu = obj.mu_;
             nu = obj.nu_;
             U = obj.U_;
             
             exph = exp(-h/2);
             exph2 = exp(-h);
             kappa = kappa-exph2;
             mu = mu-exph2*(x-mu)/kappa;
             nu = nu-1;
             U = cholupdate(U,exph*x,'-');
             
             obj.kappa_ = kappa;
             obj.mu_ = mu;
             obj.nu_ = nu;
             obj.U_ = U;
         end
         
         function y = logPredPdf(obj, X, h) 
             % predictive densiity is multivariate T(mun, Sn(Kappan+exph2)/Kappan*v)
             % v is degrees of freedom = nun-d+1 
             kappa = obj.kappa_;
             mu = obj.mu_;
             nu = obj.nu_;
             U = obj.U_;
             
             exph = exp(-h/2);
             exph2 = exp(-h);
             d = size(X,1);
             v = (nu-d+1);
             U = sqrt((1+exph2/kappa)/v)*cholupdate(U,sqrt(kappa)*mu,'-');
             
             X = bsxfun(@minus,exph.*X,exph.*mu);
             Q = U'\X;
             q = dot(Q,Q,1);  % quadratic term (M distance)
             o = -log(1+q/v)*((v+d)/2);
             c = gammaln((v+d)/2)-gammaln(v/2)-(d*log(v*pi)+2*sum(log(diag(U))))/2;
             y = c+o;
         end
     end
end
