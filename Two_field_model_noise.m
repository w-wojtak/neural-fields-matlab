%% This code simulates the neural field model from the article
%  'A dynamic neural field model of continuous input integration'
%  by W. Wojtak et al.
%
% This example is a stochastic version of the field equations with
% additive noise.
%
% (c) Weronika Wojtak, Feb 2023

%% Cleaning
clear; clc; close all

%% Spatial discretization
L = 3*pi; dx = 0.01; xDim = -L:dx:L;  N = numel(xDim);

%% Temporal discretization
T = 30; dt = 0.01; tDim = 0:dt:T; M = numel(tDim);

%% Set up functions
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
kernel = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex * exp(-0.5 * (x).^2 / s_ex^2) - ...
    A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;

%% parameters
p(1) = 1000;     % mu
p(2) = 2;        % A_ex
p(3) = 1.5;      % s_ex
p(4) = 1.0;      % A_inh
p(5) = 2.5;      % s_inh
p(6) = 0.1;      % g_i
p(7) = 0.25;     % theta
p(8) = 1;        % tau

%% Connectivity function and its FFT
w = kernel(xDim,p(2),p(3),p(4),p(5),p(6)); wHat = fft(w);

%% Noise kernel and its FFT
kernel_noise = pi * cos(xDim); wNoiseHat = fft(kernel_noise);

epsi = 0.001; % noise strength

%% Inputs
A_I = 1; sigma_I = 1; critDist = 2.25;

Input = zeros(M, N);
Input(1/dt:2/dt-1, :) = repmat(A_I * gauss(xDim-critDist, 0, sigma_I) + ...
    A_I * gauss(xDim+critDist, 0, sigma_I),1/dt,1);

%% Initial data
K = 0; u_field = p(7) - ones(1,N); v_field = K - u_field;

%% Main loop
for i = 1:M
    f = sigmoid(u_field, p(1), p(7));
    convolution = dx * ifftshift(real(ifft(fft(f) .* wHat)));
    conv_noise = sqrt(dt/1)* dx * ifftshift(real(ifft(fft(sqrt(dt)*(randn(1, N))) .* wNoiseHat)));
    
    u_field = u_field + dt/p(8) * (-u_field + convolution + v_field + Input(i, :)) + sqrt(epsi) * conv_noise;
    v_field = v_field + dt/p(8) * (-v_field - convolution + u_field);
end

%% Plot results
figure; plot(xDim,[u_field;v_field],'linewidth',2), hold on
legend('u(x)','v(x)')
plot(xDim, p(7)*ones(1,N),':k','linewidth',2);
ax = gca;
set(gca,  'FontSize', 20)
set(gca,  'XLim', [-L L])
