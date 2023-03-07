% This code simulates the recall of memorized sequence of inputs in two 
% spatial dimensions.
%
% The sequence learning is done in the file
% 'Two_field_model_2D_learn_sequence.m'
%
% For details see 'A neural integrator model for planning and value-based
% decision making of a robotics assistant' by W. Wojtak et al.
%
% (c) Weronika Wojtak, March 2023

%% Cleaning
clear; clc;

%% Space
L = 50; N = 500; dx = 2*L/N; x  = (-L+(0:N-1)*dx)'; 
[X,Y] = meshgrid(x,x); halfField = floor(N/2);

%% Time
T = 35; dt = 0.05; tspan = 0:dt:T-dt; M = numel(tspan);

%% For plots
scrsz = get(0,'ScreenSize');
plotHandleD = figure('Position',[scrsz(3)/4 scrsz(4)/2 scrsz(3)/4 scrsz(4)/4]);
plotHandleWM = figure('Position',[scrsz(3)/2 scrsz(4)/2 scrsz(3)/4 scrsz(4)/4]);

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);

%% Set up kernels
wmex = @(x,A,A_inh,sigma_exc,sigma_inh,w_inh)  A * gauss(x, 0, sigma_exc) - A_inh * gauss(x, 0, sigma_inh) - w_inh;
W_mex = wmex(sqrt(X.^2 + Y.^2),3,1.5,1,1.5,0.1); wHat_mex = fft2(W_mex); 

wosc = @(x,A,b,alpha) A*(exp(-b*abs(x)).*((b*sin(abs(alpha*x)))+cos(alpha*x)));
W = wosc(sqrt(X.^2 + Y.^2),7,0.6,1);  wHat = fft2(W);
W_d = wosc(sqrt(X.^2 + Y.^2),7,0.6,1);  wHat_d = fft2(W_d);

wlat = @(x,A,sigma,g) A*gauss(x, 0, sigma) - g;
W_lat = wlat(sqrt(X.^2 + Y.^2),3,0.3,1); wHat_lat = fft2(W_lat); 

%% parameters
beta = 1000; theta = 0.5;  tau_d = 20; tau_wm = 5;
tau_hdec = 25; h_dec = -6; hWM = 2; h_dec_in = 42;
c_d = 3; c_wm = 3; % strenght of inputs to u_d and u_wm

%% initialize fields
load('memory_gradient_2D.mat'); 

u_d = u_fieldFinal - h_dec_in;
u_wm = -hWM * ones(N,N);

% for saving the activity of bump centers in the decision field
tc2save = zeros(4,M);

%% Main loop
for i = 1:M
f_d = sigmoid(u_d,beta,theta);  fHat_d = fft2(f_d);
f_wm = sigmoid(u_wm,beta,theta);  fHat_wm = fft2(f_wm);

conv_d = (2*L/N)^2*ifftshift(real(ifft2(fHat_d .* wHat_d)));
conv_wm = (2*L/N)^2*ifftshift(real(ifft2(fHat_wm .* wHat)));               

h_dec = h_dec + (dt/tau_hdec); 
u_d = u_d + dt/tau_d * (-u_d + conv_d - c_d*conv_wm + h_dec + u_fieldFinal);
u_wm = u_wm + dt/tau_wm * (-u_wm + conv_wm + c_wm*f_d .* u_d - hWM);

if mod(i,10)==0
    disp(['t =  ',num2str(i*dt)])
    figure(plotHandleD), mesh(x,x,u_d), title('decision field')
    figure(plotHandleWM), mesh(x,x,u_wm), title('working memory field')
    drawnow; hold off
end

tc2save(1,i) = u_d(252,227);
tc2save(2,i) = u_d(127,228);
tc2save(3,i) = u_d(252,452);
tc2save(4,i) = u_d(377,152);
end

%% Plot results
% figure;
% mesh(x,x,u_wm),
% xlabel ('X','fontsize',14)

% plot time courses
figure
plot(tspan,theta*ones(1,M),'--k'), hold on
plot(tspan,tc2save), hold on
set(gca,  'YLim', [-10 10]), title('Time courses of decision  bump centers')


