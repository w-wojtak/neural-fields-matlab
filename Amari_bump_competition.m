% This code uses a forward Euler method to simulate the Amari model
% with two inputs. 
%
% The kernel w(x) is a Gaussian function. This example illustrates
% a decision process, since only the stronger of two inputs triggers 
% the evolution of a bump.
%
% (c) Weronika Wojtak, Feb 2023

%% cleaning
clear; clc;

%% spatial discretization
L = 15; dx = 0.05; xDim = -L:dx:L; N = numel(xDim);

%% temporal discretization
T = 10; dt = 0.01; tDim = 0:dt:T; M = numel(tDim);

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
w_lat = @(x,A,sigma,g_i) A * exp(-0.5 * (x).^2 / sigma^2) - g_i;

%% parameters
theta = 0.2;   % theta
beta = 1000;   % sigmoid steepness
tau = 1;       % time constant

%% set kernel
p(1) = 2;     % A
p(2) = 0.75;  % sigma
p(3) = 0.5;   % g_i
w = w_lat(xDim,p(1),p(2),p(3)); w_hat = fft(w);

%% initial data
u_field = -theta * ones(1, N);

%% inputs
A_I1 = 1; A_I2 = 0.9; sigma_I = 1; distance = 6;
Input = zeros(M, N);
I_S = A_I1 * gauss(xDim-distance, 0, sigma_I) +  A_I2 * gauss(xDim+distance, 0, sigma_I);
Input(1/dt:2/dt-1, :) = repmat(I_S,1/dt,1);

%% main loop
for i = 1:M
    f = sigmoid(u_field, beta, theta); f_hat = fft(f);
    convolution = dx * ifftshift(real(ifft(f_hat .* w_hat)));
    u_field = u_field + dt/tau * (-u_field + convolution + Input(i, :));
    
    if mod(i,10)==0
        plot(xDim,u_field,'linewidth',2), hold on
        plot(xDim,Input(i,:),'linewidth',2), legend('u(x,t)','I(x,t)')
        plot(xDim,theta*ones(1,N),'--k','linewidth',1)
        set(gca,  'XLim', [-L L]), set(gca,  'YLim', [-2 2])
        set(gca,  'FontSize', 15)
        pause(0.1); hold off
    end
    
    if mod(i,50)==0, disp(num2str(i*dt)), end
end

%% plot results
figure
plot(xDim,u_field,'k','linewidth',3), hold on
plot(xDim,theta*ones(1,N),':k','linewidth',3),
xlabel('x'); ylabel('u(x)');
set(gca,  'XLim', [-L L]), set(gca,  'FontSize', 20), hold off



