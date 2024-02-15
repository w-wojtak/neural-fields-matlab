% This function simulates a neural field model using the forward Euler method. 
% It includes an external input.
%
% The integral is computed using the trapezoidal rule, 
% employing Matlab's 'trapz' function.
%
% The kernel used in this model is the Mexican hat function, to which a spatially 
% correlated noise pattern is added.
%
% (c) Weronika Wojtak, Feb 2024

%% cleaning
clear; clc;

%% spatial discretization
L = 1*pi; dx = 0.01; xDim = -L:dx:L; N = numel(xDim);
[X, Y] = meshgrid(xDim); x_shifted = X - Y;

%% temporal discretization
T = 50; dt = 0.01; tDim = 0:dt:T; M = numel(tDim);

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
w_mex = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex * exp(-0.5 * (x).^2 / s_ex^2) - A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;

%% parameters
p(1) = 1;      % A_ex
p(2) = 0.3;    % s_ex
p(3) = 0.4;    % A_inh
p(4) = 0.5;    % s_inh
p(5) = 0.05;   % g_i

theta = 0.1;   % theta
beta = 1000;   % sigmoid steepness
tau = 1;       % time constant

%% kernel with noise
w = zeros(N, N);
epsi = 0.85; spatial_scale = 5;
kernel_noise = pi * cos(spatial_scale * xDim); wNoiseHat = fft(kernel_noise);

w_clean = w_mex(x_shifted,p(1),p(2),p(3),p(4),p(5));

for j = 1:N
    noise_j = epsi*(randn(1, N));
    conv_noise = sqrt(dt/1)* dx * ifftshift(real(ifft(fft(noise_j) .* wNoiseHat)));
    w(j, :) = w_clean(j, :) + conv_noise;

end

%% initial data
u_field =  -theta * ones(1, N);

%% inputs
A_I = 0.5; sigma_I = 0.2;
Input = zeros(M, N);
I_S = A_I * gauss(xDim, 0, sigma_I);
Input(1/dt:2/dt-1, :) = repmat(I_S,1/dt,1);

%% main loop
for i = 1:M
    f = repmat(sigmoid(u_field, beta, theta),N,1)';
    integral = trapz(xDim, f .* w);
    u_field = u_field + dt/tau * (-u_field + integral + Input(i, :));

    if mod(i,500)==0, disp(num2str(i*dt)), end
end

%% bump center position
[~, max_idx] = max(u_field);
xDim(max_idx)

%% plot results
figure
plot(xDim,I_S,'m','linewidth',3), hold on
plot(xDim,u_field,'k','linewidth',3), hold on
plot(xDim,theta*ones(1,N),':k','linewidth',3), hold on
xlabel('x'); ylabel('u(x)');
set(gca,  'XLim', [-L L]), set(gca,  'FontSize', 20), hold off
set(gca,  'YLim', [-0.1, 0.5]),
yLimits = ylim;
% Adding the vertical line at 0 to see how much the bump moved
line([0 0], yLimits, 'Color', 'b', 'LineWidth', 2); 


