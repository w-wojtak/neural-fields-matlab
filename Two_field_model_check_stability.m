% This code is used to determine the existence and stability of bump
% solutions in the neural field model from the article 'A dynamic neural 
% field model of continuous input integration' by W. Wojtak et al.
%
% (c) Weronika Wojtak, March 2023

%% cleaning
clear; clc;

%% delta range
delta = 0:0.1:10;

%% theta
theta = [];

%% Mexican-hat kernel
p(1) = 3;      % A_ex
p(2) = 2;    % s_ex
p(3) = 1.5;    % A_inh
p(4) = 4;      % s_inh
p(5) = 0.1;    % g_i
w = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex * exp(-0.5 * (x).^2 / s_ex^2) - A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;

%% set K, K=u(x,0)+v(x,0)
K = 0;

%% loop over deltas
for i = 1:numel(delta)
    theta = [theta (integral(@(x)w(x,p(1),p(2),p(3),p(4),p(5)),0,delta(i)) + K) / 2];
end

%% plot results
figure; 
plot(delta,theta)
xlabel('\Delta'), ylabel('F(\Delta)')
hold on, plot(delta,ones(length(delta))*K/2,'--k')

 