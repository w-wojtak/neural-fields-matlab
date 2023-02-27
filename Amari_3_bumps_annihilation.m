% This code uses a forward Euler method to simulate a neural field model
% with input.
%
% Two transient Gaussian inputs are applied at time t=1 at positions 
% x_{1,2,3} \in {-5.5, 0, 5.5}. 
% In the resulting solution, the middle bump is suppressed below threshold, 
% i.e. we observe bump annihilation.
%
% See Fig. 16 in 'A dynamic neural field model of continuous input integration'
%  by W. Wojtak et al. (2021).
%
% (c) Weronika Wojtak, Feb 2023

%% cleaning
clear; clc;

%% spatial discretization
L = 7*pi; N = 2^12; dx = 2*L/N; xDim = (-L+(0:N-1)*dx);

%% temporal discretization
T = 20; dt = 0.01; tDim = 0:dt:T; M = numel(tDim);

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
w_mex = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex * exp(-0.5 * (x).^2 / s_ex^2) - A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;

%% parameters
p(1) = 2;      % A_ex
p(2) = 1.25;   % s_ex
p(3) = 1;      % A_inh
p(4) = 2.5;    % s_inh
p(5) = 0.1;    % g_i

theta = 0.4;   % theta
beta = 1000;   % sigmoid steepness
tau = 1;       % time constant

%% initial data
u_field = -theta * ones(1, N);
history_u = zeros(M, N);

%% connectivity function
w = w_mex(xDim,p(1),p(2),p(3),p(4),p(5)); w_hat = fft(w);

%% inputs
crit_dist = 5.5; % distance of the input centers from x=0
A_I = 10.0; sigma_I = 1.0;
Input = zeros(M, N);
I_S1 = A_I * gauss(xDim, 0, sigma_I);
I_S2 = A_I * gauss(xDim, crit_dist, sigma_I);
I_S3 = A_I * gauss(xDim, -crit_dist, sigma_I);
Input(1/dt:2/dt, :)       = repmat((I_S1 + I_S2 + I_S3),1+(1/dt),1);

%% main loop
for i = 1:M
    f = sigmoid(u_field, beta, theta); f_hat = fft(f);
    convolution = dx * ifftshift(real(ifft(f_hat .* w_hat)));
    u_field = u_field + dt/tau * (-u_field + convolution + Input(i, :));
    history_u(i,:) = u_field;
end

%% PLOT RESULTS
% heat map
figure
imagesc(flipud(history_u)), colormap hot, colorbar
xlabel('x'); ylabel('t','Rotation',0);
ax = gca; set(gca,  'FontSize', 30);
ax.XTick = [186 N/2 3912];
ax.XTickLabel = ({-20,0,20});
ax.YTick = [1 M];
ax.YTickLabel = ({T, 0});

% input and final state of u-field
figure
plot(xDim,I_S1+I_S2+I_S3,'Color',[1 1 1].*0.7,'linewidth',4), hold on
plot(xDim,u_field,'k','linewidth',3), hold on
plot(xDim,theta*ones(1,N),':k','linewidth',2), 
xlabel('x'); ylabel('u(x), I(x)');
set(gca,  'XLim', [-20 20])
ax = gca; set(gca,  'FontSize', 30)

