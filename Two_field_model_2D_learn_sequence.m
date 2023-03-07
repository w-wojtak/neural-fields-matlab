% This code uses a forward Euler method to simulate the two field model
% with inputs in two spatial dimensions.
%
% An activation gradient established in the two field model encodes serial
% order of memorized inputs.
%
% The recall of learned sequence is done in the file
% 'Two_field_model_2D_recall_sequence.m'
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
T = 50; dt = 0.1; tspan = 0:dt:T-dt; M = numel(tspan);

%% Initialize fields
u_field = -1 * ones(N,N,'single'); v_field = 0.25 - u_field;

%% utils
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
gauss = @(x,mu,sigma) exp(-0.5 * (x-mu).^2 / sigma^2);

%% Set up kernel
wmex = @(x,A,A_inh,sigma_exc,sigma_inh,w_inh)  A * gauss(x, 0, sigma_exc) - A_inh * gauss(x, 0, sigma_inh) - w_inh;
W2d = wmex(sqrt(X.^2 + Y.^2),7,4,1.5,2.5,0.0);  wHat2d = fft2(W2d);

%% Parameters
beta = 1000; theta = 0.3; tau_u = 1;

%% Input in time
targets = [50, 45;
            25, 45;
            50, 90;
            75, 30];

gaussian2d = gauss(sqrt(X.^2 + Y.^2),0,1);
Input = zeros(N,N,M,'single');
Input1 = zeros(N,N); Input2 = zeros(N,N); Input3 = zeros(N,N); Input4 = zeros(N,N);

%% Constant input
Input_const = 1;

%% Transient inputs
A_input = 5.0; % input amplitude

Input_temp = gaussian2d(halfField-10/dx:halfField+10/dx, halfField-10/dx:halfField+10/dx);

Input1((targets(1,1)-10)/dx+1:(targets(1,1)+10)/dx+1,(targets(1,2)-10)/dx+1:(targets(1,2)+10)/dx+1) = A_input*Input_temp;
Input2((targets(2,1)-10)/dx+1:(targets(2,1)+10)/dx+1,(targets(2,2)-10)/dx+1:(targets(2,2)+10)/dx+1) = A_input*Input_temp;
Input3((targets(3,1)-10)/dx:(targets(3,1)+10)/dx,(targets(3,2)-10)/dx:(targets(3,2)+10)/dx) = A_input*Input_temp;
Input4((targets(4,1)-10)/dx:(targets(4,1)+10)/dx,(targets(4,2)-10)/dx:(targets(4,2)+10)/dx) = A_input*Input_temp;

inpLength = 2/dt + 1;

Input(:,:,floor(4/dt):floor(6/dt)) = repmat(Input1,[1 1 inpLength]);
Input(:,:,floor(12/dt):floor(14/dt)) = repmat(Input2,[1 1 inpLength]);
Input(:,:,floor(25/dt):floor(27/dt)) = repmat(Input3,[1 1 inpLength]);
Input(:,:,floor(39/dt):floor(41/dt)) = repmat(Input4,[1 1 inpLength]);

% for saving the activity of bump centers in the u-field
tc2save = zeros(4,M);

%% Main loop
for i = 1:M
    
    f = sigmoid(u_field,beta,theta);  fHat = fft2(f);
    convolution = (2*L/N)^2*ifftshift(real(ifft2(wHat2d .* fHat)));
    u_field = u_field + dt/tau_u * (-u_field + convolution + v_field + Input(:,:,i) + f.*Input_const);
    v_field = v_field + dt/tau_u * (-v_field - convolution + u_field);
    
    % if mod(i,10)==0
    %     disp(['t =  ',num2str(i*dt)])
    %     mesh(x,x,u_field),
    %     drawnow; hold off
    % end
    
    tc2save(1,i) = u_field(252,227);
    tc2save(2,i) = u_field(127,228);
    tc2save(3,i) = u_field(252,452);
    tc2save(4,i) = u_field(377,152);
end

% save sequence memory
u_fieldFinal = u_field;
save('memory_gradient_2D.mat','u_fieldFinal');

%% Plot results
% plot end state of memory field
figure;
mesh(x,x,u_field), axis square, view(3)
ylabel ('y'), xlabel ('x'), zlabel ('u(x,y)')
set(get(gca,'ZLabel'),'Rotation',0)
set(gca,  'FontSize', 20), hold off

% plot time courses
figure;
plot(tspan,theta*ones(1,M),'--k'), hold on
plot(tspan,tc2save), hold on
title('Time courses of memory  bump centers')


