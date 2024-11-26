% This code implements a traveling wave solution in the Amari model
% using a forward Euler method for simulation with input.
%
%
% (c) Weronika Wojtak, Nov 2024

%% cleaning
clear; clc;

%% spatial discretization
L = 25; dx = 0.05; xDim = -L:dx:L; N = numel(xDim);

%% temporal discretization
T = 20; dt = 0.01; tDim = 0:dt:T; M = numel(tDim);

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
w_lat = @(x,A,sigma,g_i) A * exp(-0.5 * (x).^2 / sigma^2) - g_i;
w_mex = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex * exp(-0.5 * (x).^2 / s_ex^2) - A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;
w_osc =  @(x,A,b,alpha) A*(exp(-b*abs(x)).*((b*sin(abs(alpha*x)))+cos(alpha*x)));

%% parameters
theta = 0.5;   % theta
beta = 1000;   % sigmoid steepness
tau = 1;       % time constant

%% Mexican-hat kernel
c = 15; % strength of asymmetry

p(1) = 3;      % A_ex
p(2) = 1.5;    % s_ex
p(3) = 1.5;    % A_inh
p(4) = 3;      % s_inh
p(5) = 0.2;    % g_i
w_sym = w_mex(xDim,p(1),p(2),p(3),p(4),p(5)); 
w = w_sym -c * diff([w_sym w_sym(end)]);
w_hat = fft(w);

%% initial data
u_field = -theta * ones(1, N);

%% inputs
A_I = 3; sigma_I = 1;
Input = zeros(M, N);
I_S = A_I * gauss(xDim, 0, sigma_I);
Input(1/dt:2/dt-1, :) = repmat(I_S,1/dt,1);

%% main loop
for i = 1:M
    f = sigmoid(u_field, beta, theta); f_hat = fft(f);
    convolution = dx * ifftshift(real(ifft(f_hat .* w_hat)));
    u_field = u_field + dt/tau * (-u_field + convolution + Input(i, :));
    
    if mod(i,50)==0, disp(num2str(i*dt)), end
end

%% plot results
figure
plot(xDim,u_field,'k','linewidth',3), hold on
plot(xDim,theta*ones(1,N),':k','linewidth',3),
xlabel('x'); ylabel('u(x)');
set(gca,  'XLim', [-L L]), set(gca,  'FontSize', 20), hold off



