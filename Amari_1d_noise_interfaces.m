% This code simulates the Amari equation in 1d with the additive noise,
% starting from the stationary solution as the initial condition.
%
% It tracks the bump boundaries using the interface equations from 
% Section 4.3 in
% Krishnan, N., Poll, D. B., & Kilpatrick, Z. P. (2018). Synaptic efficacy
% shapes resource limitations in working memory. Journal of computational 
% neuroscience, 44, 273-295.
% 
% (c) Weronika Wojtak, June 2023


%% Cleaning
clear; clc;

%% Functions
sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));
kernel = @(x,A_ex,s_ex,A_in,s_in,g_i) A_ex* exp(-0.5 * (x).^2 / s_ex^2) - A_in * exp(-0.5 * (x).^2 / s_in^2) - g_i;
stability_amari = @(delta,kernel,p) integral(@(x)kernel(x,p(2),p(3),p(4),p(5),p(6)),0,delta) - p(7);

%% Parameters 
p(1) = 100;         % beta
p(2) = 3;           % A_ex   
p(3) = 1.4;         % s_ex   
p(4) = 1.5;         % A_inh
p(5) = 3.5;         % s_inh
p(6) = 0.2;         % g_i
p(7) = 0.0;         % theta          

L = 3*pi; N = 2^11; dx = 2*L/N; xDim = (-L+(0:N-1)*dx); 
dt = 0.05; tspan = 0:dt:20; M = numel(tspan);

tau_u = 1;

%% Get the steady state U(x) from theory
problem = @(delta) stability_amari(delta,kernel,p);
delta_theory = fzero(problem,4);

U = zeros(N,1);

for i = 1:length(xDim)
U(i) = integral(@(x)kernel(x,p(2),p(3),p(4),p(5),p(6)),0, xDim(i) + delta_theory/2) -...
    integral(@(x)kernel(x,p(2),p(3),p(4),p(5),p(6)),0, xDim(i)-delta_theory/2);
end

%% Kernel and its fft
w = kernel(xDim,p(2),p(3),p(4),p(5),p(6)); wHat = fft(w);

%% Noise kernel and its fft
eps = 0.2;
w_c = (5*pi)/L;
kernel_noise = cos(xDim * w_c);
wHat_noise = fft(kernel_noise);

%% Get the spatial gradient at the interface points of the steady state (alpha_bar and -alpha_bar)
w_0 = kernel(0,p(2),p(3),p(4),p(5),p(6));
w_delta = kernel(delta_theory,p(2),p(3),p(4),p(5),p(6));
alpha_bar = (w_0 - w_delta);

%% for storage
x_1 = zeros(1,M); x_2 = zeros(1,M);
dif_x1 = zeros(1,M); dif_x2 = zeros(1,M);
history_u = zeros(M,N);

xDim_1 = zeros(1,M);
xDim_2 = zeros(1,M);
xDim_3 = zeros(1,M);

%% Main loop

% initial state
u_field = U';

for i = 1:M

    f = sigmoid(u_field, p(1), p(7)); fHat = fft(f);
    conv = dx * ifftshift(real(ifft(fHat .* wHat)));

    % noise for this time step
    noise_i = sqrt(dt)*(randn(1, N)); noiseHat = fft(noise_i);
    conv_noise =  dx * ifftshift(real(ifft(noiseHat .* wHat_noise)));

    u_field = u_field + dt/tau_u * (-u_field + conv) + sqrt(eps) * conv_noise;

    % get the bump interfaces
    for j = 2:N
        if u_field(j-1) <= p(7) && u_field(j) > p(7)
            x_1(i) = j;
        end

        if u_field(j-1) >= p(7) && u_field(j) < p(7)
            x_2(i) = j;
        end
    end

    % get the bump shift due to noise
    if i > 1      
        dif_x1(i) = (x_1(i)- x_1(i-1))*dx;      
        dif_x2(i) = (x_2(i)- x_2(i-1))*dx;       
    end

    % save history
    history_u(i,:)  = u_field;

end


%% The evolution of the interfaces and bump center

delta = delta_theory;

c1 = 0; % at time t=0 the bump is cetered at x=0
x1 = xDim(x_1(1));
x2 = xDim(x_2(1));

for i = 1:M

    % get the bump center and the boundary points x1 and x2
    int_x1_x2 = integral(@(x)kernel(x,p(2),p(3),p(4),p(5),p(6)),0,delta);

    c1 = c1 + ((1/(2*alpha_bar)) * (dif_x2(i) + dif_x1(i)));
    x1 = x1 + ((-1/alpha_bar) * (dt * (-p(7) + int_x1_x2) -  dif_x1(i)));
    x2 = x2 + ((1/alpha_bar) * (dt * (-p(7) + int_x1_x2)  +  dif_x2(i)));

    % get the indices of the points above
    [ ~, ind1 ] = min( abs( xDim-x1 ) );
    [ ~, ind2 ] = min( abs( xDim-x2 ) );
    [ ~, ind3 ] = min( abs( xDim-c1 ) );

    xDim_1(i) = ind1;
    xDim_2(i) = ind2;
    xDim_3(i) = ind3;

end


%% PLOT RESULTS
figure
imagesc((history_u)), title ('u(x)'), hold on
colormap hot

plot((xDim_1),linspace(0,size(tspan,2),size(tspan,2)),'c','linewidth',3), hold on
plot((xDim_2),linspace(0,size(tspan,2),size(tspan,2)),'c','linewidth',3), hold on
plot((xDim_3),linspace(0,size(tspan,2),size(tspan,2)),'m','linewidth',3), hold on

