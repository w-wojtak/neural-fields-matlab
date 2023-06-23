% This code turns the input amplitude into bump position.
% It uses the two-filed integrator model to integrate the input and Amari
% model to encode the traveling wave and read-out.
% 
% (c) Weronika Wojtak, June 2023

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
theta_readout = 1;
offset_wave = 0.2; A_wave = 0.75; sigma_wave = 0.5;
c_space = 2.7;

%% set kernels
w_time = w_mex(xDim-(L/2),1.5,0.4,0.75,0.6,0.1); w_hat_time = fft(w_time);
w_space = w_mex(xDim-(L/2),2.5,0.5,1,0.8,0.1); w_hat_sp = fft(w_space);

%% Initial data
u_space = h_space * ones(1, N); 
K = 0.0; u_field =  -theta * ones(1, N); v_field = K - u_field;

flag_1 = 0; flag_2 = 0;

%% Input for the integrator
A_rate = 15;
u_rate = A_rate * gauss(xDim-(L/2),0,0.4);
Input = (1/A_rate) * gauss(xDim-(L/2),0,0.4);

% for plots
f_time = figure; f_time.Position = [300 100 1000 800];

%% Main loop
for i = 1:M
    
    u_wave = A_wave * gauss(xDim, (i*dt)-offset_wave, sigma_wave);
    
    f_time = sigmoid(u_field, beta, theta); f_hat_time = fft(f_time);
    conv_time = dx * ifftshift(real(ifft(f_hat_time .* w_hat_time)));
    
    f_space = sigmoid(u_space, beta, theta); f_hat_space = fft(f_space);
    conv_space = dx * ifftshift(real(ifft(f_hat_space .* w_hat_sp)));

    u_space = u_space + dt/tau_u * (-u_space + conv_space + h_space + ...
        c_space * flag_1 * u_wave * (1-flag_2));
    
    u_field = u_field + dt/tau * (-u_field + conv_time + v_field + Input);
    v_field = v_field + dt/tau * (-v_field - conv_time + u_field);
    
    uv_sum = u_field + v_field;
    
    flag_1 = max(uv_sum > theta_readout);
    flag_2 = max(uv_sum > theta_readout + 0.05);
    
    if mod(i,50)==0, 
        subplot(221)
        plot(xDim,u_rate,'linewidth',2), set(gca,'YLim',[0 16])
        title('Input to integrate (amplitude)')
        subplot(222)
        plot(xDim,uv_sum,'linewidth',2), set(gca,'YLim',[-2 6]), hold on
        plot(xDim,theta_readout*ones(1,N),'r','linewidth',2), 
        title('Integrator model (u-field)'), hold off
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