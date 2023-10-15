%% This code simulates the model from
%  'A dynamic neural field model of continuous input integration'
%  by W. Wojtak et al.
%  but instead using +u and +v, it has +f(u) and +f(v),
%  where f() is a piece-wise linear function and with negative feedback, 
%  i.e. the signs of u, v, and conv are changed.

clear; clc
%% Spatial coordinates
L = 4*pi; N = 2^10; dx = 2*L/N; xDim = (-L+(0:N-1)*dx);

%% Temporal coordinates
dt = .01; tspan = 0:dt:50; M = numel(tspan);

%% Functions
kernel = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex* exp(-0.5 * (x).^2 / s_ex^2) - A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;
gauss = @(x,mu,sigma) exp(-0.5*(x-mu).^2/sigma^2);
sigmoid = @(x,beta,theta)  1./ (1 + exp(-beta*(x-theta)));
pwl = @(x,theta) (x.*(x>0)).*(x<theta) + theta*(x>=theta);

%% Paramaters
p(1) = 1000;   % beta
p(2) = 1.3;      % A_ex
p(3) = 1.2;   % s_ex
p(4) = 0.8;      % A_inh
p(5) = 2.2;    % s_inh
p(6) = 0.05;    % w_inh
p(7) = 0.2;    % theta

beta = p(1); theta = p(7); tau = 1;

% threshold for pwl function
kappa = 1.5;

%% Initial data
K = 0.0;
u_field =  -theta * ones(1, N); v_field = K - u_field;
history_u = zeros(M, N);
history_v = zeros(M, N);

%% Connectivity function
w = kernel(xDim,p(2),p(3),p(4),p(5),p(6)); wHat = fft(w);

%% Input
A_I = 4; sigma_I = 1;
Input = zeros(M, N);
Input_pattern = A_I * gauss(xDim, 0, sigma_I);
Input(1/dt:2/dt, :) = repmat(Input_pattern,1+(1/dt),1);

%% Main loop
for i = 1:M
    f = sigmoid(u_field, beta, theta);
    convolution = dx * ifftshift(real(ifft(fft(f) .* wHat)));
    % 2-field model:
    u_field = u_field + dt/tau * (-u_field + convolution - pwl(v_field, kappa) + Input(i, :));
    v_field = v_field + dt/tau * (-v_field + convolution - pwl(u_field, kappa));
    history_u(i,:) = u_field; history_v(i,:) = v_field;
end

%% Plot results
figure
plot(xDim,u_field,'k','linewidth',3), hold on
plot(xDim,v_field,'--k','linewidth',3), hold on
% plot(xDim,u_field+v_field,'-m','linewidth',4), hold on
plot(xDim,theta*ones(1,N),':k','linewidth',2);
xlabel('x'); ylabel('u(x), v(x)');
ax = gca;
set(gca,  'FontSize', 20)
set(gca,  'XLim', [-L L])

