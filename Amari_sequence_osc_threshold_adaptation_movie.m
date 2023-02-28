% This code uses a forward Euler method to simulate the Amari model
% with a sequence of three inputs.
%
% The kernel is an oscillatory function. State-dependent threshold
% adaptation is used to control the amplitude of individual bumps.
% As a result bump's height increases over time, and at the end of the
% experiment we observe an activation gradient with bump amplitudes
% decreasing as a function of elapsed time since sequence onset.
%
% (c) Weronika Wojtak, Feb 2023

%% cleaning
clear; clc;

%% spatial discretization
L = 60; dx = 0.05; xDim = -L:dx:L; N = numel(xDim);

%% temporal discretization
T = 70; dt = 0.01; tDim = 0:dt:T; M = numel(tDim);

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
w_osc =  @(x,A,b,alpha) A*(exp(-b*abs(x)).*((b*sin(abs(alpha*x)))+cos(alpha*x)));

%% parameters
p(1) = 1;      % A
p(2) = 0.5;    % b
p(3) = 0.9;    % alpha

theta = 1;     % theta
beta = 1000;   % sigmoid steepness
tau_u = 1;     % time constant of the field
tau_h = 30;    % time constant of the threshold adaptation

%% set kernel
w = w_osc(xDim,p(1),p(2),p(3)); w_hat = fft(w);

%% initial data
u_field = -theta * ones(1, N);
h_u = zeros(1, N);
history_u = zeros(M, N);

%% inputs
crit_dist = 35; A_I = 3; sigma_I = 1.5;

Input = zeros(M, N);
I_S1 = A_I * gauss(xDim, -crit_dist, sigma_I);
I_S2 = A_I * gauss(xDim, 0, sigma_I);
I_S3 = A_I * gauss(xDim, crit_dist, sigma_I);

Input(10/dt:11/dt, :) = repmat((I_S1),1+(2/dt-1/dt),1);
Input(18/dt:19/dt, :) = repmat((I_S2),1+(2/dt-1/dt),1);
Input(31/dt:32/dt, :) = repmat((I_S3),1+(2/dt-1/dt),1);

%% open video file
vidObj = VideoWriter('Amari_sequence_h_adaptation.avi');
vidObj.Quality = 100;
vidObj.FrameRate = 10;
open(vidObj);

%% main loop
for i = 1:M
    f = sigmoid(u_field, beta, theta); f_hat = fft(f);
    convolution = dx * ifftshift(real(ifft(f_hat .* w_hat)));
    h_u = h_u + dt/tau_h * f; % threshold adaptation
    u_field = u_field + dt/tau_u * (-u_field + convolution + Input(i, :) + h_u);
    history_u(i,:) = u_field;
    
    if mod(i,50)==0
        plot(xDim,u_field,'linewidth',2), hold on
        plot(xDim,Input(i,:),'linewidth',2),
        plot(xDim,h_u,'linewidth',2), legend('u(x,t)','I(x,t)','h_u (x)')
        plot(xDim,theta*ones(1,N),'--k','linewidth',1)
        set(gca,  'XLim', [-L L]), set(gca,  'YLim', [-1 5])
        xlabel('x'); set(gca,  'FontSize', 15)
        pause(0.1); hold off
        % write to file
        writeVideo(vidObj, getframe(gcf));
    end
    
    if mod(i,500)==0, disp(num2str(i*dt)), end
end

% close video file
close(gcf)
close(vidObj);


%% plot results
% figure
% plot(xDim,u_field,'k','linewidth',3), hold on
% plot(xDim,h_u,'c','linewidth',2), hold on
% legend('u(x)','h_u (x)');
% plot(xDim,theta*ones(1,N),':k','linewidth',3),
% xlabel('x');
% set(gca,  'XLim', [-L L]), set(gca,  'FontSize', 20), hold off

