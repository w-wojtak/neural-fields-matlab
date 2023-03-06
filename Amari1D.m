% This function is passed to MATLAB's ode45 solver in the file
% 'Amari_ode45.m' used to simulate a neural field model.
%
% The spatial convolution of w and f is computed using a fast Fourier
% transform (FFT).

function [F] = Amari1D(u,p,wHat,sigmoid,N,L)

beta = p(1);
theta = p(7);

%% Firing rate function
f = sigmoid(u,beta,theta); fHat = fft(f);

%% Convolution
conv = (2*L/N)*ifftshift(real(ifft(fHat .* wHat)));

%% Right-hand side
F = - u + conv;
F = F(:);

end
