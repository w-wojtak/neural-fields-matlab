function [f,df] = ComputeFiringRate(u,mu,h)

%    f  = 1./(1+exp( -mu*(u-h) ));
%    if nargout > 1
%      df = mu*f(u).*(1-f(u));
%    end

%    f   = (u > h) .* exp( -mu./( (u-h).^2) );
%    if nargout > 1
%      df  = 2*mu*f(u)./(u-h).^3;
%    end
   
   f   = heaviside(u - h) .* exp( -mu./( (u-h).^2) );
  if nargout > 1
  df = dirac(u-h).*  exp( -mu./( (u-h).^2) )+2*mu*f(u)./(u-h).^3.*heaviside(u - h);
  end

end
