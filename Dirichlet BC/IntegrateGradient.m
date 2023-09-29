function [v,xi] = IntegrateGradient(u,h,x,v1);

  %% Rename parameters
  N = size(u,1);
  v = zeros(size(u));
  dx      = x(2) - x(1);

  for i = 1:N
    v(i) = v1; 
    if i >= 2
      kappa = ones(i,1); kappa(1) = 0.5; kappa(i) = 0.5; kappa = kappa*dx;
      v(i) = v(i) + kappa'*u(1:i);
    end
  end
      y1= h*ones(N,1); y2= v; 
      [xi, yi] = polyxpoly(x,y1,x,y2);
      
end
