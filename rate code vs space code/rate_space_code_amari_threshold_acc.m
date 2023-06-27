% This code turns the input amplitude into bump position.
% It uses the Amari model, threshold accommodation and traveling wave
% mechanism.
% 
% (c) Weronika Wojtak, June 2023
%
% TODO: Add a working memory field u_wm to inhibit the bump in the decision
% field u_acc.

%% Cleaning
clear; clc

%% spatial discretization
L = 20; dx = 0.01; xDim = 0:dx:L; N = numel(xDim);

%% temporal discretization
T = 20; dt = 0.01; tDim = 0:dt:T; M = numel(tDim);

%% Functions
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
w_mex = @(x,A_ex,s_ex,A_in,s_in,g_i)  A_ex* exp(-0.5 * (x).^2 / s_ex^2) -...
    A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;

%% Paramaters
beta = 1000; theta = 0; tau = 1;
h_space = -0.4;
tau_u = 1;
theta_readout = 0;
offset_wave = 0.09; A_wave = 0.75; sigma_wave = 0.5;
c_space = 7;

h_d_init = 19;
tau_h_acc = 1;

%% set kernels
w_time = w_mex(xDim-(L/2),1.5,0.4,0.75,0.6,0.1); w_hat_time = fft(w_time);
w_space = w_mex(xDim-(L/2),2.5,0.5,1,0.8,0.1); w_hat_sp = fft(w_space);

%% Input from learning
A_rate = 5;
u_rate = A_rate * gauss(xDim-(L/2),0,0.5);

%% Initial data
u_space = h_space * ones(1, N); 
u_acc = u_rate - h_d_init;

h_acc = -h_d_init;

flag_1 = 0; flag_2 = 0;

% for plots
f = figure; f.Position = [300 100 1000 800];

%% Main loop
for i = 1:M
    
    u_wave = A_wave * gauss(xDim, L - (i*dt) + offset_wave, sigma_wave);
    
    f_acc = sigmoid(u_acc, beta, theta); f_hat_acc = fft(f_acc);
    conv_time = dx * ifftshift(real(ifft(f_hat_acc .* w_hat_time)));
    
    f_space = sigmoid(u_space, beta, theta); f_hat_space = fft(f_space);
    conv_space = dx * ifftshift(real(ifft(f_hat_space .* w_hat_sp)));

    u_space = u_space + dt/tau_u * (-u_space + conv_space + h_space + ...
        c_space * flag_1 * u_wave * (1-flag_2));
    
    h_acc =  h_acc + dt/tau_h_acc;
    u_acc = u_acc + dt/tau * (-u_acc + conv_time + u_rate + h_acc);

    flag_1 = max(u_acc > theta_readout);
    flag_2 = max(u_acc > theta_readout + 0.2);
    
    if mod(i,50)==0, 
        subplot(221)
        plot(xDim,u_rate,'linewidth',2), set(gca,'YLim',[0 16])
        title('Input bump (amplitude)')
        subplot(222)
        plot(xDim,u_acc,'linewidth',2), set(gca,'YLim',[-20 20]), hold on
        plot(xDim,theta_readout*ones(1,N),'r','linewidth',2), 
        title('Amari model + threshold accommodation'), hold off
        subplot(223)
        plot(xDim,u_wave,'linewidth',2), set(gca,'YLim',[0 1]),
        title('Traveling wave u_{wave}')
        subplot(224)
        plot(xDim,u_space,'linewidth',2), set(gca,'YLim',[-1 1]),
        title('Read-out field (position)')
        drawnow; pause(0.1), hold off
        disp(num2str(i*dt))
    end
    
end

%% Get the bump position in u_sp field
[~,max_sp] = max(u_space); 

disp(' ')
disp(['Bump amplitude: ', num2str(A_rate)])
disp(['Bump position: ', num2str(xDim(max_sp))])
disp(' ')