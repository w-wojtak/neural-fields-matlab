% This code reproduces result from Fig. 3b in
% "Nonlinear analysis of breathing pulses in a synaptically coupled neural
% network", Folias, S. E. (2011)
%
% (c) Weronika Wojtak, Mar 2023

%% cleaning
clear; clc;

%% spatial discretization
L = 2*pi; N = 2^9; dx = 2*L/N; xDim = (-L+(0:N-1)*dx);

%% temporal discretization
T = 80; dt = 0.01; tspan = 0:dt:T-dt; M = numel(tspan);

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);
kernel = @(x,we,se) (we/(sqrt(pi)*se)) .* exp(-(x/se).^2);

%% parameters
beta = 1000;
theta = 0.375;
ro = 2.75;
nu = 0.1;
we = 1; se = 1;
tau = 1;

%% initial condition
u_0 = 0.7 * gauss(xDim,0,0.7) - 0.1;
q_0 = zeros(1,N);
u = u_0; q = q_0;
hist_u = zeros(M,N); hist_q = zeros(M,N);

%% kernel
W = kernel(xDim,we,se);  wHat = fft(W);

%% input
I = 1.9 * gauss(xDim,0,1.2);

% %% open video file
% vidObj = VideoWriter('Amari_breather.avi');
% vidObj.Quality = 100;
% vidObj.FrameRate = 5;
% open(vidObj);

%% solution
for i = 1:M
    f = sigmoid(u,beta,theta); fHat = fft(f);
    convolution = dx * ifftshift(real(ifft(fHat .* wHat)));
    
    u = u + dt/tau * (-u -ro * q + convolution + I);
    q = q + dt * nu *( -q + u);
    
%     if mod(i,100)==0
%         plot(xDim,u,'linewidth',2), hold on
%         plot(xDim,theta*ones(1,N),':k','linewidth',2)
%         set(gca,'YLim', [-0.25 1]), set(gca,'XLim', [-L L]), 
%         set(gca, 'YTick', [0 1])
%         title(['t = ' num2str(i*dt)])
%         set(gca,'FontSize', 20), set(0,'defaulttextInterpreter','tex') 
%         xlabel('x'); ylabel('u(x)'); pause(0.1); hold off;
%         % write to file
%         writeVideo(vidObj, getframe(gcf));
%     end
    
    hist_u(i,:) = u; hist_q(i,:) = q;
    
end

% % close video file
% close(gcf)
% close(vidObj);

figure
imagesc(hist_u), colorbar
set(gca,'XTick',[53 256 461])
set(gca,'XTickLabel',[-5 0 5])
xlabel('x'); ylabel('t');
set(gca,'FontSize', 20), 
% 
% figure,
% set(0,'defaultTextInterpreter', 'latex');
% set(0,'DefaultAxesFontName', 'Helvetica ')
% plot(xDim,u,'linewidth',2), hold on
% plot(xDim,u_0,'--'), legend('u field',  'initial u'),hold on
% plot(xDim,theta*ones(1,N),':')
