% This code implements the mapping of timing information to spatial 
% position by generating a traveling bump in the u_wave field and reading 
% its thresholded activity into the u_read field.
%
% (c) Weronika Wojtak, Nov 2024

%% cleaning
clear; clc;

%% spatial discretization
L = 30; dx = 0.05; xDim = -L:dx:L; N = numel(xDim);

%% temporal discretization
T = 25; dt = 0.01; tDim = 0:dt:T; M = numel(tDim);

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

threshold_read = 3;

%% Mexican-hat kernel
c = 15; % strength of asymmetry

p(1) = 3;      % A_ex
p(2) = 1.5;    % s_ex
p(3) = 1.5;    % A_inh
p(4) = 3;      % s_inh
p(5) = 0.2;    % g_i
w_sym = w_mex(xDim,p(1),p(2),p(3),p(4),p(5)); 
w_hat_sym = fft(w_sym);

w = w_sym -c * diff([w_sym w_sym(end)]);
w_hat = fft(w);

%% initial data
u_wave = -theta * ones(1, N);

u_read = -theta * ones(1, N);

%% inputs
A_I = 2; sigma_I = 1;
Input_gauss = zeros(M, N);
I_S = A_I * gauss(xDim, -10, sigma_I);
Input_gauss(1/dt:2/dt-1, :) = repmat(I_S,1/dt,1);

Input_flat = zeros(M, N);
Input_flat(10/dt:11/dt,:) = 1;

figure;

%% main loop
for i = 1:M
    Input = Input_gauss(i, :) + Input_flat(i, :);

    f_read = sigmoid(u_read, beta, theta); f_hat_read = fft(f_read);
    convolution_read = dx * ifftshift(real(ifft(f_hat_read .* w_hat_sym)));
    
    f = sigmoid(u_wave, beta, theta); f_hat = fft(f);
    convolution = dx * ifftshift(real(ifft(f_hat .* w_hat)));
   
    u_wave = u_wave + dt/tau * (-u_wave + convolution + Input);

    u_read = u_read + dt/tau * (-u_read + convolution_read + u_wave.*(u_wave > threshold_read));

    
    if mod(i,20)==0
        disp(num2str(i*dt)),
        subplot(211)
        plot(xDim,Input_flat(i, :),'linewidth',2), hold on
        plot(xDim,u_wave,'linewidth',2), hold off
        set(gca,  'YLim', [-3 3.5]), 
        set(gca,  'XLim', [-L L]), 
        subplot(212)
        plot(xDim,u_read,'linewidth',2), 
        set(gca,  'XLim', [-L L]), 
        pause(0.1)
    end
end

%% plot results
figure
plot(xDim,Input, 'Color',[0.7, 0.7, 0.7],'linewidth',4), hold on
plot(xDim,u_wave,'k','linewidth',3), hold on
plot(xDim,theta*ones(1,N),':k','linewidth',3),
xlabel('x'); ylabel('u_{wave}(x), I(x)');
set(gca,  'XLim', [-L L]), 
set(gca,  'YLim', [-3 3.5]), 
set(gca,  'FontSize', 20), hold off

figure
plot(xDim,u_read,'k','linewidth',3), hold on
plot(xDim,theta*ones(1,N),':k','linewidth',3),
xlabel('x'); ylabel('u_{read}(x)');
set(gca,  'XLim', [-L L]), 
set(gca,  'YLim', [-3 3.5]), 
set(gca,  'FontSize', 20), hold off
