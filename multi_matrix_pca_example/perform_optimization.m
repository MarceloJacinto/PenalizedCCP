% MIT License
%
% Copyright (c) 2022 Marcelo Jacinto
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
%
function [steps,avg_time,objective_val] = perform_optimization(m, n, p, tau_0, mu, tau_max, delta, initialize_best)

% Initiate the matrices
Ai = cell(p,1);

% Generate a diagonal matrix with entries drawn uniformly from the interval [0,1]
A = diag(rand([n, 1]));
q = orth(randn(n,n));

% Apply the orthogonal transformation to A
A = q' * A * q;

% Generate p matrices Ai
i = 1;
while i <= p

    % Vary the parameters of A by a random factor of 50%
    Aux = ((1.0*rand() - 0.5) * A) + A;
    
    % Check if the matrix is positive definite by looking at the eigen values 
    % (this is valid because the matrix is symmetric)
    % otherwise, we should generate a new matrix instead
    if all(eig(Aux) > 0)
        
        % Then get the highest eigen value of each matrix
        % and subtract the value to the matrix to get a negative definite
        % matrix
        Aux = Aux - ((max(eig(Aux)) + 1.0) * eye(n));
        Ai{i} = Aux;
        i = i + 1;
    end    
end

% Initialize the X matrix
if initialize_best == 1
    % Initialize using the 'best' criteria

    Xi = cell(p,1);

    % For each matrix Ai
    for i=1:p
        % Compute the m principal components of each Ai matrix
        aux = pca(Ai{i});
        Xi{i} = aux(:,1:m);     
    end

    % Compute the Xi that yields the best objective value
    objective_values = zeros(p,1);
    aux = zeros(p,1);
    for k=1:p
        % Compute the trace for each Xi' * Ai * Xi
        for i=1:p
            aux(i) = trace(Xi{k}' * Ai{i} * Xi{k});
        end
    
        % Compute the minimum of the trace
        objective_values(k) = -min(aux);
    end

    % Get the index that produced the maximized the objective function
    [~, index] = min(objective_values);

    % Use the corresponding X0 as the initial value to the problem
    X0 = Xi{index};
    
% Use the average criteria
else
    % Compute the m principal components of the average of the Ai matrices
    sum_matrix = zeros(n,n);
    for i=1:p
        sum_matrix = sum_matrix + Ai{i};
    end
    X0 = pca(reshape(sum_matrix / p, n, n));
    X0 = X0(:,1:m);
    
end

% Initialize the iterative problem
tau_k = tau_0 * eye(m, m);
X_k = X0;

tau_k_old = tau_k;
X_k_old = X0;
s1_old = zeros(m, m);
s2_old = zeros(m, m);

k = 0; % Number of iterations
stop = false;

step_time = 0;

while ~stop

    % Solve the convexified optimization problem
    cvx_begin sdp

        % Declare the optimization variable
        variable X(n,m);
        variable aux_obj(p,1);
        
        % Declare the slack variables
        variable s1(m,m);
        variable s2(m,m);

        % Compute the trace for every -tr(X' * Ai * X)
        objective = 0.0;
        for i=1:p
            X_A_X = 0.0;
            for j=1:m
                X_A_X = X_A_X + X(:,j)' * Ai{i} * X(:,j);
            end
            objective_vector(i) = -trace(X_A_X);
        end

        % Define the optmization problem
        minimize max(objective_vector) + trace(tau_k'*s1) + trace(tau_k'*s2)
        subject to
            [s1 + eye(m,m), X'; X, eye(n,n)] >= 0;
            s2 - eye(m,m) + X_k'*X_k + 2*X_k'*(X-X_k) >=0;
            s1 >= 0;
            s2 >= 0;
    cvx_end
    
    % Clear the objective vector (because CVX will freak out every
    % iteration for changing the type from double to cvx var)
    clear objective_vector;
    
    % Update the values of Xk
    X_k_old = X_k;
    X_k = X;
    
    % Update tk if this condition is met
    if norm(mu * tau_k, 2) <= tau_max
           tau_k_old = tau_k;
           tau_k = mu * tau_k;
    end
    
    % Update the iteration
    k = k + 1;
    step_time = step_time + cvx_cputime;
    
    % ---------------------------
    % Check the stopping criteria
    % ---------------------------
    [stop, objective_value, diff] = stop_criteria(X_k_old, X_k, s1_old, s2_old, s1, s2, tau_k_old, tau_k, Ai, p, delta);
    s1_old = s1;
    s2_old = s2;
    
    fprintf('Objetive function value: %d \n', objective_value);
    fprintf('Difference criteria: %d \n', diff);
    fprintf('Iteration: %d \n', k);
end

steps = k;
avg_time = step_time / steps;
objective_val = cvx_optval;

end
