% This code uses a forward Euler method to simulate the two field model
% with a sequence of three inputs.
%
% The kernel is a mexican hat function.
% 
% For details see 'A dynamic neural field model of continuous input 
% integration' by W. Wojtak et al.
%
% (c) Weronika Wojtak, March 2023

%% cleaning
clear; clc;

%% spatial discretization
L = 10*pi; dx = 0.05; xDim = -L:dx:L; N = numel(xDim);

%% temporal discretization
T = 50; dt = 0.01; tDim = 0:dt:T; M = numel(tDim);

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
K = 0.0;
u_field = -theta * ones(1, N);
v_field = K - u_field;
history_u = zeros(M, N); history_v = zeros(M, N);

%% kernel
w = w_mex(xDim,p(1),p(2),p(3),p(4),p(5)); w_hat = fft(w);

%% inputs
crit_dist = 18.0; A_I = 1; sigma_I = 1;

Input = zeros(M, N);
I_S1 = A_I * gauss(xDim, -crit_dist, sigma_I);
I_S2 = A_I * gauss(xDim, 0, sigma_I);
I_S3 = A_I * gauss(xDim, crit_dist, sigma_I);

Input(1/dt:2/dt, :) = repmat((I_S1),1+(1/dt),1);
Input(10/dt:11/dt, :) = repmat((I_S2),1+(1/dt),1);
Input(20/dt:21/dt, :) = repmat((I_S3),1+(1/dt),1);

%% main loop
for i = 1:M
    f = sigmoid(u_field, beta, theta); f_hat = fft(f);
    convolution = dx * ifftshift(real(ifft(f_hat .* w_hat)));
    % 2-field model:
    u_field = u_field + dt/tau * (-u_field + convolution + v_field + Input(i, :));
    v_field = v_field + dt/tau * (-v_field - convolution + u_field);
    history_u(i,:) = u_field; history_v(i,:) = v_field;
end

%% plot results
figure
plot(xDim,u_field,'k','linewidth',3), hold on
plot(xDim,v_field,'--k','linewidth',3), hold on
plot(xDim,theta*ones(1,N),':k','linewidth',2),
xlabel('x'); ylabel('u(x),v(x)');
set(gca,  'XLim', [-L L]), set(gca,  'FontSize', 20), hold off

