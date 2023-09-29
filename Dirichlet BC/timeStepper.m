%% Cleaning
clear all, close all, clc;

%% Spatial coordinates
L  = 40*pi; N = 1*(2^8+1); dx = 2*L/(N-1); xDim  = (-L+(0:N-1)*dx)'; 

%% Parameters
p0(1) = L;             % L
p0(2) = N;             % N
p0(3) = 0;             % mu      steepness (firing rate)   
p0(4) = 0.5;           % theta       threshold
p0(5) = 0.1;           % u(-L)   BC

%% Differentiation matrix
e = ones(N,1);
Dx = spdiags([-e e],[-1 1],N,N);
Dx = Dx/(2*dx);
Bx = Dx; Bx(1,1) = 1; Bx(1,2) = 0;

%%  Initial data 
u_0 = exp((-0.05*(xDim+0).^2)./2); 
u_0(1)=p0(5);

%% Differentiate initial data
u0 = Dx*u_0;   

%% Connectivity function 
wf = @(z,A,B) A/sqrt(pi*B) * exp(-z.^2/B);
a1 = 14; a2 = 13; b1 = 3000/128; b2 = 3000/20;
synr = @(z) wf(z,a1,b1) - wf(z,a2,b2);
k=@(z,A,B) A/sqrt(pi*B) *(-2.*z/B).* exp(-z.^2/B);
dsynr = @(x) k(x,a1,b1)-k(x,a2,b2);

%% Plot kernel
% figure; 
% plot(xDim,synr(xDim),'k.-',xDim,Dx*synr(xDim),'g.-',xDim,dsynr(xDim),'m.-');
% drawnow;

%% Connectivity matrix
tic; disp('Computing connectivity matrix...'); 
for i=1:N
    for j=1:N
      W(i,j) = dsynr(xDim(i)-xDim(j));
    end
end
toc;

%% Quadrature weights
rho  = ones(N,1); rho(1)=0.5; rho(end)=0.5; rho=rho*dx;

%% Problem handle
problemHandle = @(t,u) IntegralModel1DGradient(u,p0,W,rho,xDim);

%% Run
tic; disp('Calculating...'); 
[t,U] = ode45(problemHandle,0:0.01:50,u0);

for i=1:length(t)
z        = U(i,:)';
[u_field, xi]   = IntegrateGradient(U(i,:)',p0(4),xDim,p0(5));
end

z = U(end,:)';
[u_field, xi]   = IntegrateGradient(U(end,:)',p0(4),xDim,p0(5));
toc;
%% Plot space time
% figure;
% [X,T]= meshgrid(x,t);
% surf(X,T,U); shading interp;
% xlabel('x','FontSize',18);ylabel('t','FontSize',18);zlabel('u','FontSize',18);

%% Plot result
figure
plot(xDim,u_field,'linewidth',2), hold on
plot(xDim,u_0,'linewidth',2), hold on
legend('u(x,T)','u(x,0)');
plot(xDim,p0(4)*ones(1,N),':k','linewidth',2),
xlabel('x'); ylabel('u(x)');
set(gca,  'XLim', [-L L]), 
set(gca,  'FontSize', 20), hold off

