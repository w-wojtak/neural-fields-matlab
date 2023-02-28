% This code is used to determine the existence and stability of a bump
% solution in the Amari model.
%
% The existence of a solution of width ? is determined by the roots of
% W(?) = ?.
%
% The stability condition is that a steady state of width ? is stable if
% W'(?) < 0, and unstable otherwise.
%
% For details see Amari, S. I. (1977). Dynamics of pattern formation in 
% lateral-inhibition type neural fields. Biological cybernetics,
% 27(2), 77-87.
%
% (c) Weronika Wojtak, Feb 2023

%% cleaning
clear; clc;

%% delta range
delta = 0:0.1:10;

% theta
theta = [];

%% choose kernel

% Gaussian
p(1) = 1.5;    % A
p(2) = 1;      % sigma
p(3) = 0.1;    % g_i
w = @(x,A,sigma,g_i) A * exp(-0.5 * (x).^2 / sigma^2) - g_i;

for i = 1:numel(delta)
    k = delta(i);
    theta_i = integral(@(x)w(x,p(1),p(2),p(3)),0,k);
    theta = [theta theta_i];
end


% Mexican-hat
% p(1) = 3;      % A_ex
% p(2) = 1.5;    % s_ex
% p(3) = 1.5;    % A_inh
% p(4) = 3;      % s_inh
% p(5) = 0.2;    % g_i
% w = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex * exp(-0.5 * (x).^2 / s_ex^2) - A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;
% 
% for i = 1:numel(delta)
%     k = delta(i);
%     theta_i = integral(@(x)w(x,p(1),p(2),p(3),p(4),p(5)),0,k);
%     theta = [theta theta_i];
% end


% Oscillatory
% p(1) = 1;      % A
% p(2) = 0.3;    % b
% p(3) = 0.9;    % alpha
% w =  @(x,A,b,alpha) A*(exp(-b*abs(x)).*((b*sin(abs(alpha*x)))+cos(alpha*x)));
%
% for i = 1:numel(delta)
%     k = delta(i);
%     theta_i = integral(@(x)w(x,p(1),p(2),p(3)),0,k);
%     theta = [theta theta_i];
% end


%% plot results
figure; 
plot(delta,theta,'-k','linewidth',2)
xlabel('\Delta'), ylabel('W(\Delta)')
% check bump existence for a particular value of \theta
theta_value = 1;
hold on, plot(delta,ones(length(delta),1)*theta_value,'--k','linewidth',1)


