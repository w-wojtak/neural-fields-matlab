% This code plots an equilibrium solution $U(x)= \lim\limits_{t\to \infty}
% (u(x,t))$ of the Amari equation.
%
% For details see Amari, S. I. (1977). Dynamics of pattern formation in
% lateral-inhibition type neural fields. Biological cybernetics,
% 27(2), 77-87.
%
% (c) Weronika Wojtak, Feb 2023

%% cleaning
clear; clc;

%% functions
kernel = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex* exp(-0.5 * (x).^2 / s_ex^2) - A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;
amari_stab = @(delta,kernel,p,theta) integral(@(x)kernel(x,p(1),p(2),p(3),p(4),p(5)),0,delta) - theta;

%% parameters
p(1) = 3;      % A_ex
p(2) = 1.5;    % s_ex
p(3) = 1.4;    % A_inh
p(4) = 3;      % s_inh
p(5) = 0.2;    % g_i

theta  = 0.5;  % theta

%% find the bump width delta by solving for theta = W(delta)
delta_guess = 4;
problem = @(delta) amari_stab(delta,kernel,p,theta);
delta = fzero(problem,delta_guess);

%% spatial discretization
L = 20; N = 2^10; dx = 2*L/N; x = (-L+(0:N-1)*dx);

%% calculate U(x)
U = zeros(N,1);
for i = 1:length(x)
    U(i) = integral(@(x)kernel(x,p(1),p(2),p(3),p(4),p(5)),0, x(i)+delta/2) - integral(@(x)kernel(x,p(1),p(2),p(3),p(4),p(5)),0, x(i)-delta/2);
end

%% plot results
figure
plot(x,U,'k','linewidth',3), hold on
plot(x,theta*ones(1,N),':k','linewidth',3),
xlabel('x'); ylabel('u(x)');
set(gca,  'XLim', [-L L]), set(gca,  'FontSize', 20), hold off

%% compare delta from theory with bump width
fprintf(['Delta from theory: ', num2str(delta)])
fprintf(['\nBump width: ', num2str(sum(U>theta)*dx), '\n'])

