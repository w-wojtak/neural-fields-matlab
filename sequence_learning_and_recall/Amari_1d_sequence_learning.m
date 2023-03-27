% This code implements sequence learning in the Amari model with threshold
% accommodation. The recall is done in Amari_1d_sequence_recall.m file.
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

%% parameters
p(1) = 0;      % theta
p(2) = 1000;   % sigmoid steepness
p(3) = 1;      % time constant of the field
p(4) = 30;     % time constant of the threshold adaptation

theta = p(1);    
beta = p(2);   
tau_u = p(3);     
tau_h = p(4);   

A = 1;          % A
b = 0.5;        % b
alpha = 0.9;    % alpha

%% set kernel
w = w_osc(xDim,A,b,alpha); w_hat = fft(w);

%% initial data
u_field = -1 * ones(1, N);
h_u = -1 * ones(1, N);
history_u = zeros(M, N);

%% inputs
crit_dist = 25; A_I = 3; sigma_I = 1.5;

Input = zeros(M, N);
I_S1 = A_I * gauss(xDim, 0, sigma_I);
I_S2 = A_I * gauss(xDim, -crit_dist, sigma_I);
I_S3 = A_I * gauss(xDim, crit_dist, sigma_I);

Input(3/dt:4/dt, :) = repmat((I_S1),1+(2/dt-1/dt),1);
Input(11/dt:12/dt, :) = repmat((I_S2),1+(2/dt-1/dt),1);
Input(26/dt:27/dt, :) = repmat((I_S3),1+(2/dt-1/dt),1);

%% for storing time courses
tc_1 = zeros(1, M); tc_2 = zeros(1, M); tc_3 = zeros(1, M);

%% open video file
% vidObj = VideoWriter('sequence_learning.avi');
% vidObj.Quality = 100;
% vidObj.FrameRate = 20;
% open(vidObj);

%% main loop
for i = 1:M
    f = sigmoid(u_field, beta, theta); f_hat = fft(f);
    convolution = dx * ifftshift(real(ifft(f_hat .* w_hat)));
    h_u = h_u + dt/tau_h * f; % threshold adaptation
    u_field = u_field + dt/tau_u * (-u_field + convolution + Input(i, :) + h_u);
    history_u(i,:) = u_field;
    
    if mod(i,10)==0
        plot(xDim,u_field,'linewidth',2), hold on
        plot(xDim,h_u,'linewidth',2), hold on
        plot(xDim,Input(i,:),'linewidth',2), 
        legend('u(x,t)','h_u(x,t)','I(x,t)')
        legend('Location','northeastoutside')
        plot(xDim,theta*ones(1,N),'--k','linewidth',1)
        set(gca,'XLim',[-L L]), set(gca,'YLim',[-1.5 3])
        xlabel('x'); set(gca,'FontSize', 15)
        title(['Memory field']), set(0,'defaulttextInterpreter','tex') 
        pause(0.01); hold off
        % write to file
%         writeVideo(vidObj, getframe(gcf));
    end
    
    if mod(i,100)==0, disp(num2str(i*dt)), end
    
    tc_1(i) = u_field(151); tc_2(i) = u_field(400); tc_3(i) = u_field(651);
end

%% close video file
% close(gcf)
% close(vidObj);

%% save results
save('sequence_learning_data.mat','u_field','p');

%% plot results
figure
plot(xDim,u_field,'k','linewidth',3), hold on
plot(xDim,h_u,'c','linewidth',2), hold on
legend('u(x)','h_u (x)'); 
legend('Location','northeastoutside')
plot(xDim,theta*ones(1,N),':k','linewidth',3),
xlabel('x');
set(gca,'XLim',[-L L]), set(gca,'FontSize',20), hold off

figure
plot(tDim,[tc_1; tc_2; tc_3],'linewidth',3), hold on
plot(tDim,theta*zeros(1,M),':k','linewidth',3), hold on
set(gca, 'YLim', [-0.5 3]), set(gca,'XLim',[0 T]),
xlabel('time'); ylabel('activation u_{mem}');
set(gca,'FontSize',20), hold off
