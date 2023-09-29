function [F,Jv] = IntegralModel1DGradient(u,p,W,rho,x)

  %% Rename parameters 
  N       = p(2);
  mu      = p(3);
  h       = p(4);
  v1      = p(5);
  
  %% Compute u from v
  v = IntegrateGradient(u,h,x,v1);
  
  %% Firing rates
  S = ComputeFiringRate(v,mu,h);
  
  %% Compute right-hand side
  F = -u + W* (S.*rho);      

end
