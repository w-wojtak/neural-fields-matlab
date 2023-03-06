% This code uses MATLAB's ode45 solver to simulate a neural field model
% with a gaussian as a initial condition.
%
% The problem equations can be found in the file 'Amari1D.m'

%% cleaning
clear; clc;

%% spatial discretization
L = 6*pi; N = 2^10; dx = 2*L/N; xDim = (-L+(0:N-1)*dx)';

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
w_mex = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex * exp(-0.5 * (x).^2 / s_ex^2) - A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;

%% parameters
p0(1) = 1000;     % beta
p0(2) = 2;        % A_ex
p0(3) = 1.25;     % s_ex
p0(4) = 1;        % A_inh
p0(5) = 2.5;      % s_inh
p0(6) = 0.1;      % g_i
p0(7) = 0.5;      % theta

%% initial condition
A_I = 1; sigma_I = 1;
u0 = A_I * gauss((xDim), 0, sigma_I);

%% kernel
w = w_mex(xDim,p0(2),p0(3),p0(4),p0(5),p0(6)); wHat = fft(w);

%% solution
problem = @(t,u) Amari1D(u,p0,wHat,sigmoid,N,L);
[T,U] = ode45(problem,[0 50],u0);
uFinal = (U(end,:));

%% plot results
figure
plot(xDim,uFinal,'-','linewidth',3), hold on
plot(xDim,p0(7)*ones(1,N),':k','linewidth',2),
xlabel('x'); ylabel('u(x)');
set(gca,  'XLim', [-L L]), set(gca,  'FontSize', 20), hold off



