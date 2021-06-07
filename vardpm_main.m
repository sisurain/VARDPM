% univariate case with sv + dpm
clear; clc;
%load 'USCPI.csv';
%y = USCPI;
dataraw = importdata('uslcap.csv');
data = dataraw.data(:,2);
y = 100*(data(2:end)-data(1:end-1))./data(1:end-1);
%y = 400*log(data(2:end)./data(1:end-1));
%y = reshape(y(4:end),5,2025/5);
%y = sum(y)';
T = length(y);

burnin = 200;
nloop = 2200;
%% prior
dlt0 = .97; Vdlt = .1^2;
nusig2 = 10; Ssig2 = .2^2*(nusig2-1);

mu0 = 0; kappa0 = 10; nu0 = 10; S0 = 10;
prior = GaussWishartSV(kappa0,mu0,nu0,S0);
model_name = 'DPMM';
%warning('off')
%% initialize the Markov chain

dlt = .98; signu2 = .2^2;
h = sqrt(signu2)*randn(1,T);
hh = h';
exph = exp(h);
alpha = 5; k = 0;
X = y'; [d,~]=size(X);
[z,Theta,w] = oneDPM(X,h,alpha,prior); nk = T*w; llh = zeros(nloop,1);

%{
Pi = ones(1,K)/K;
lam2K = gamrnd(nu0,1/S0,K,1); etaK = mu + sqrt(tau*lam2K).*randn(K,1);
lam2 = zK*lam2K; eta = zK*etaK;
%}
%% initialize for storeage
store_h = zeros(nloop - burnin,T);
%store_etaK = zeros(nloop - burnin,K);
%store_lam2K = zeros(nloop - burnin,K);

%store_Pi = zeros(nloop - burnin,K);
store_z = zeros(nloop - burnin,T);
store_phi = zeros(nloop - burnin,4);

%% compute a few things outside the loop
Hdlt = speye(T) - sparse(2:T,1:(T-1),dlt*ones(1,T-1),T,T);
newnusig2 = T/2 + nusig2;

counth = 0;
disp(['Starting MCMC for ' model_name '.... ']);
start_time = clock;
for loop = 1:nloop
   
    %% sample assignment parameters z
    for i = randperm(T)
        x = X(:,i);
        hi = h(i);
        k = z(i);
        Theta{k} = Theta{k}.delSample(x,hi);
        nk(k) = nk(k)-1;
        if nk(k) == 0           % remove empty cluster
            Theta(k) = [];
            nk(k) = [];
            which = z>k;
            z(which) = z(which)-1;
        end
        Pk = log(nk)+cellfun(@(t) t.logPredPdf(x,hi), Theta);
        P0 = log(alpha)+prior.logPredPdf(x,hi);
        p = [Pk,P0];
        llh(loop) = llh(loop)+sum(p-log(T));
        p = p - max(p,[],2);
        p = exp(p);
        p = p./sum(p,2);
    
        k = randmn(p);
        if k == numel(Theta)+1                 % add extra cluster
            Theta{k} = prior.clone.addSample(x,hi);
            nk = [nk,1];
        else
            Theta{k} = Theta{k}.addSample(x,hi);
            nk(k) = nk(k)+1;
        end
        z(i) = k;
    end
    [C,ia,ic] = unique(z);
    K = length(C);
    z_counts = accumarray(ic,1);
    value_counts = [C', z_counts];
    ot = sum(z_counts < T*0.01); % number of small clusters
    
    m = cell2mat(cellfun(@(t) t.mu_, Theta, 'UniformOutput', false));
    kappa = cell2mat(cellfun(@(t) t.kappa_, Theta, 'UniformOutput', false));
    nu = cell2mat(cellfun(@(t) t.nu_, Theta, 'UniformOutput', false));
    U = cell2mat(cellfun(@(t) t.U_, Theta, 'UniformOutput', false));
    
    lam2K = []; etaK = [];
    for i = 1:K
        Ui = U(:,d*(i-1)+1:d*i);
        kappai = kappa(i);
        mui = m(:,i);
        nui = nu(i);
        S = Ui'*Ui - kappai*mui*mui';
        cholS = chol(S);
        %invcholS = cholS\speye(d);
        %lam2K(i) = wishrnd(invcholS*invcholS',nui,invcholS);
        lam2K(i) = iwishrnd(S,nui);
        etaK(i) = mui + chol(lam2K(i)/kappai)'*randn(d,1);
    end
    
    zK = sparse([1:T]',z',ones(T,1),T,K); 
    lam2 = zK*lam2K'; eta = zK*etaK';         
    
    etalam = eta.*sqrt(lam2);
    alp = zeros(T,1);
    
    %% sample h
    yl = y.*sqrt(lam2); 
    HinvSH = Hdlt'*sparse(1:T,1:T,[(1-dlt^2)/signu2; 1/signu2*ones(T-1,1)])*Hdlt;
    deltah = zeros(T,1);
    HinvSHdeltah = HinvSH*deltah;
    s2 = (yl-etalam).^2;
    errh = 1; ht = hh;
    while errh> 10^(-3)
        expht = exp(ht);
        sinvexpht = s2./expht;
        alp2expht = alp.^2.*expht;
        fh = -.5 + .5*sinvexpht - .5*alp2expht;
        Gh = .5*sinvexpht + .5*alp2expht;
        Kh = HinvSH + spdiags(Gh,0,T,T);
        newht = Kh\(fh+Gh.*ht+HinvSHdeltah);
        errh = max(abs(newht-ht));
        ht = newht;          
    end 
    cholHh = chol(Kh,'lower');
    % AR-step:     
    hstar = ht;
    uh = hstar-deltah;
    logc = -.5*uh'*HinvSH*uh -.5*sum(hstar) + ...
        - .5*exp(-hstar)'*(yl-etalam-alp.*exp(hstar)).^2 + log(3);
    flag = 0;
    while flag == 0
        hc = ht + cholHh'\randn(T,1);
        vhc = hc-ht;
        uhc = hc-deltah;
        alpARc = -.5*uhc'*HinvSH*uhc -.5*sum(hc) + ...
            -.5*exp(-hc)'*(yl-etalam-alp.*exp(hc)).^2 + .5*vhc'*Kh*vhc - logc;            
        if alpARc > log(rand)
            flag = 1;
        end
    end        
    % MH-step
    vh = hh-ht;
    uh = hh-deltah;
    alpAR = -.5*uh'*HinvSH*uh -.5*sum(hh) + ...
        -.5*exp(-hh)'*(yl-etalam-alp.*exp(hh)).^2 + .5*vh'*Kh*vh - logc;
    if alpAR < 0
        alpMH = 1;
    elseif alpARc < 0
        alpMH = - alpAR;
    else
        alpMH = alpARc - alpAR;
    end    
    if alpMH > log(rand) || loop == 1
        hh = hc;
        exph = exp(hh);
        counth = counth + 1;
    end
    h = hh';
    
    %% sample dlt signu2
    % sample signu2
    errh = [hh(1)*sqrt(1-dlt^2); hh(2:end)-dlt*hh(1:end-1)];    
    newSsig2 = Ssig2 + sum(errh.^2)/2;    
    signu2 = 1/gamrnd(newnusig2, 1./newSsig2); 
    % sample dlt
    Vdlthat = 1/(1/Vdlt + hh(1:end-1)'*hh(1:end-1)/signu2);
    dlthat = Vdlthat*(dlt0/Vdlt + hh(1:end-1)'*hh(2:end)/signu2);
    dlt = tnormrnd(dlthat,Vdlthat,-1,1);
    
    
    if loop>burnin
        i = loop-burnin;
        store_h(i,:) = h';
        store_z(i,:) = z;
        store_phi(i,:) =  [K dlt signu2 ot];  
    end    
    if (mod(loop,10) == 0)
        disp([num2str(loop) ' loops... '])
    end 
end
clc;
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );

thetahat = mean(store_phi)'
hhat = mean(store_h)';

hCI = quantile(store_h,[.05 .95])';
thetaCI = quantile(store_phi,[.05 .95])';
    
    
    
    