% This function is used in the file Amari_fourier_series.m that simulates the Amari 
% model using a finite Fourier series expansion.

function F = AmariFourier(t,a,M,K,x,dx,theta,L)

    % define firing rate function
    sigmoid = @(x,beta,theta) 1 ./ (1 + exp(-beta*(x-theta)));

    % initialize 
    integral = zeros(1,M);
    u_A = zeros(M,numel(x));
    
    % recover u(x) from coefficients a_j
    u_A(1,:) = a(1)/1;
    for j =2:M     
      u_A(j,:) = a(j)*cos((pi*(j-1)*x)/L);    
    end
    sum_A = sum(u_A);
    
    % firing rate function
    beta = 100;
    f_recovered_sum = sigmoid(sum_A,beta,theta);

    % integral
    for k=1:M
     integral(k) = dx*trapz(cos((pi*(k-1)*x)/L).*f_recovered_sum); 
    end
     
    % right-hand side
    F = -a + K .* integral';
end







