% This code uses a forward Euler method to simulate the two field model
% with input in two spatial dimensions.
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
L = 20; N = 2^9; h = 2*L/N; x  = (-L+(0:N-1)*h)';
[X,Y] = meshgrid(x,x);

%% temporal discretization
T = 20; dt = 0.01; tspan = 0:dt:T-dt; M = numel(tspan);

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
w_mex = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex * exp(-0.5 * (x).^2 / s_ex^2) - A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;

%% parameters
p(1) = 3;      % A_ex
p(2) = 1;      % s_ex
p(3) = 1.2;    % A_inh
p(4) = 1.6;    % s_inh
p(5) = 0.2;    % g_i

theta = 0;     % theta
beta = 1000;   % sigmoid steepness
tau = 1;       % time constant

%% initial data
K = -0.5; u0 = -0.5 * ones(N,N);  v0 = K - u0;
u = u0; v = v0;

%% set kernel
W2d = w_mex(sqrt(X.^2 + Y.^2),p(1),p(2),p(3),p(4),p(5));  wHat = fft2(W2d);

%% input
I_0 = zeros(N, N);
A_I = 3; sigma_I = 1;
I_S = A_I * gauss(sqrt(X.^2 + Y.^2), 0, sigma_I);
t_start = 1;
t_stop = 5;

%% for plotting
plot_activity = 0;

if plot_activity == 1
    hFig = figure;
    set(hFig, 'Position', [900 300 570 510]);
end

%% main loop
for i = 1:M
    
    if i > t_start/dt && i < t_stop/dt
        Input = I_S;
    else
        Input = I_0;
    end
    
    f = sigmoid(u,beta,theta); fHat = fft2(f);
    convolution = (2*L/N)^2*ifftshift(real(ifft2(fHat .* wHat)));
    
    u = u + dt/tau * (-u + v + convolution + Input);
    v = v + dt/tau * (-v + u - convolution);
    
    if plot_activity == 1 && mod(i,100) == 0
        surf(X,Y,u); shading interp; view(3); axis square; axis tight;
        colormap(jet)
        title(['t =  ' num2str(i*dt)])
        drawnow;
    end
    
    if mod(i,100) == 0
        disp(['t = ' num2str(i*dt)])
    end
    
end

%% plot results
figure
surf(X,Y,u); shading interp; view(3); axis square; axis tight;
colormap(jet)
xlabel('x'); ylabel('y'); zlabel('u(x,y)');

