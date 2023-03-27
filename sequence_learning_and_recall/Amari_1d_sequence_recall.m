% This code recalls the sequence of inputs learned in 
% Amari_1d_sequence_learning.m file.
%
% For details see "Rapid learning of complex sequences with time
% constraints: A dynamic neural field model." by Ferreira, Flora, et al.
% (2020)
%
% (c) Weronika Wojtak, Mar 2023

%% cleaning
clear; clc;

%% spatial discretization
L = 40; dx = 0.1; xDim = -L:dx:L; N = numel(xDim);

%% temporal discretization
T = 40; dt = 0.01; tDim = 0:dt:T; M = numel(tDim);

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
w_osc =  @(x,A,b,alpha) A*(exp(-b*abs(x)).*((b*sin(abs(alpha*x)))+cos(alpha*x)));
w_lat = @(x,A,sigma,g) A*gauss(x, 0, sigma) - g;

%% load data from learning
load('sequence_learning_data.mat')

%% rename parameters
theta = p(1);    
beta = p(2);   
tau_u = p(3);     
tau_h = p(4);    

%% remaining parameters
h_d_init = 2.66;
tau_h_dec = tau_h;
h_wm = -1.5;
c_wm = 4; % strength of inhibitory connections from u_wm to u_dec

A_wm = 1.5; b_wm = 0.5; alpha_wm = 0.8;
A_dec = 1.5; sigma_dec = 0.75; g_dec = 0.1;

%% set kernels
w_wm = w_osc(xDim,A_wm,b_wm,alpha_wm); w_hat_wm = fft(w_wm);
w_dec = w_lat(xDim,A_dec,sigma_dec,g_dec); w_hat_dec = fft(w_dec); 

%% initial data
u_mem = u_field;
u_dec = u_field - h_d_init;

h_dec = -h_d_init * ones(1, N);
u_wm = h_wm * ones(1, N);

%% for storing time courses
tc_1 = zeros(1, M); tc_2 = zeros(1, M); tc_3 = zeros(1, M);

%% open video file
% vidObj = VideoWriter('sequence_recall.avi');
% vidObj.Quality = 100;
% vidObj.FrameRate = 20;
% open(vidObj);

f = figure; f.Position = [500 500 900 400];

%% main loop
for i = 1:M
    f_dec = sigmoid(u_dec, beta, theta); f_hat_dec = fft(f_dec);
    f_wm = sigmoid(u_wm, beta, theta); f_hat_wm = fft(f_wm);
        
    conv_dec = dx * ifftshift(real(ifft(f_hat_dec .* w_hat_dec)));
    conv_wm = dx * ifftshift(real(ifft(f_hat_wm .* w_hat_wm)));
    
    h_dec =  h_dec + dt/tau_h_dec;
    
    u_dec = u_dec + dt/tau_u * (-u_dec + conv_dec + h_dec + u_mem - c_wm * f_wm .* u_wm);
    u_wm = u_wm + dt/tau_u * (-u_wm + conv_wm + h_wm + f_dec .* u_dec);
    
    if mod(i,10)==0
        subplot(121)
        plot(xDim,u_dec,'linewidth',2), hold on
        plot(xDim,theta*ones(1,N),'--k','linewidth',1)
        set(gca,'XLim',[-L L]), set(gca,'YLim', [-4 2])
        title('Decision field')
        xlabel('x'); ylabel('u_{dec}(x)'); 
        set(gca,'FontSize', 20), hold off
        subplot(122)
        plot(xDim,u_wm,'linewidth',2), hold on
        plot(xDim,theta*ones(1,N),'--k','linewidth',1)
        set(gca,'XLim',[-L L]), set(gca,'YLim', [-2 3])
        title('Working memory field')
        xlabel('x'); ylabel('u_{wm}(x)'); 
        set(gca,'FontSize', 20),hold off
        pause(0.01);
        % write to file
%         writeVideo(vidObj, getframe(gcf));
    end

    tc_1(i) = u_dec(151); tc_2(i) = u_dec(400); tc_3(i) = u_dec(651);
     
    if mod(i,100)==0, disp(num2str(i*dt)), end
end

%% close video file
% close(gcf)
% close(vidObj);

% %% plot results
% figure
% plot(xDim,u_field,'k','linewidth',3), hold on
% plot(xDim,h_u,'c','linewidth',2), hold on
% legend('u(x)','h_u (x)');
% plot(xDim,theta*ones(1,N),':k','linewidth',3),
% xlabel('x');
% set(gca,'XLim',[-L L]), set(gca,  'FontSize', 20), hold off

figure
plot(tDim,[tc_1; tc_2; tc_3],'linewidth',3), hold on
plot(tDim,theta*zeros(1,M),':k','linewidth',3), hold on
set(gca,'YLim',[-0.5 2.3]), set(gca, 'XLim', [0 T]),
xlabel('time'); ylabel('activation u_{dec}');
set(gca,'FontSize',20), hold off

