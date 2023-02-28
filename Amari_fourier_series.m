% This code simulates the Amari model using a finite Fourier series
% expansion.
% 
% We expand u(x) and the kernel w(x) in Fourier series, substitute the two
% series in the Amari equation, and solve the resulting system of ODEs
% using MATLAB's ode45 function.
%
% (c) Weronika Wojtak, Feb 2023

%% cleaning
clear; clc;

%% spatial discretization
L = 60; dx = 0.01; xDim = -L:dx:L; N = numel(xDim);

%% kernel
w_osc =  @(x,A,b,alpha) A*(exp(-b*abs(x)).*((b*sin(abs(alpha*x)))+cos(alpha*x)));

%% parameters
theta = 3;       % threshold for f(u)
M = 20;  % no. of Fourier modes

%% initial condition
u_field = -3 + 10 * exp(-xDim.^2/18);
A = zeros(M,1);
K = zeros(M,1);

%% Fourier coefficients for the initial profile
a_0 = dx*(1/L)*(trapz(u_field))/2;       
for j =1:M        
  A(j) = dx*(1/L)*(trapz(u_field.*cos((pi*j*xDim)/L)));
end
u_A = zeros(M,numel(xDim));
for j =1:M     
  u_A(j,:) = A(j)*cos((pi*j*xDim)/L);    
end
recovered_sum = a_0 + sum(u_A); % recovered sum, just to check if ok

%% Fourier coefficients for the weight function
p(1) = 2;      % A
p(2) = 0.08;   % b
p(3) = 0.3;    % alpha
w = w_osc(xDim,p(1),p(2),p(3));

k_0 = dx*(1/L)*(trapz(w))/2;
for j =1:M-1        
  K(j)=dx*(1/L)*(trapz(w.*cos((pi*j*xDim)/L)));
end
u_K = zeros(M,numel(xDim));
for j =1:M-1     
  u_K(j,:) = K(j)*cos((pi*j*xDim)/L);    
end
recovered_kernel = k_0 + sum(u_K);  % recovered sum, just to check if ok

%% ploting sums to verify if ok
% subplot(1,2,1), plot(xDim,u_field), hold on
% plot(xDim,recovered_sum),legend('initial profile','recovered sum')
% subplot(1,2,2), plot(xDim,w), hold on
% plot(xDim,recovered_kernel),legend('kernel','recovered kernel')

%% Solution
aInit = [a_0; A(1:M-1)];
K = [k_0; K(1:M-1)];
tspan = [0 100];
problemHandle = @(t,a) AmariFourier(t,a,M,K,xDim,dx,theta,L); 
% Solve:
[T,U] = ode45(problemHandle,tspan,aInit);

%% recover u(x) from obtained coefficients
A_final = U(end,:);
u_Sum = zeros(M,numel(xDim));
u_Sum(1,:) = A_final(1)/2;
for j =2:M
  u_Sum(j,:) = A_final(j)*cos((pi*(j-1)*xDim)/L);    
end
recovered_U = sum(u_Sum);

%% plot the result
figure
set(0,'DefaultAxesFontName', 'Helvetica ')
set(0,'DefaultAxesFontSize', 20)
plot(xDim,u_field,'--','linewidth',2), hold on
plot(xDim,recovered_U,'-','linewidth',2),legend('initial profile','solution'),hold on
plot(xDim,theta*ones(numel(xDim),1),':','linewidth',1), hold on
xlabel('x'); ylabel('u(x)'); 
set(gca, 'XLim', [-L L]), set(gca,  'FontSize', 20), hold off

