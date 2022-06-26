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
function [stop, objective_next, diff] = stop_criteria(Xk_prev, Xk_next, s1_prev, s2_prev, s1_next, s2_next, tau_k_prev, tau_k_next, Ai, p, delta)

% Alocating memory for better performance
obj_vector_prev = zeros(p,1);
obj_vector_next = zeros(p,1);

for i=1:p
    % Computing the previous objective function value
    obj_vector_prev(i) = -trace(Xk_prev' * Ai{i} * Xk_prev); 
    
    % Compute the next objective function value
    obj_vector_next(i) = -trace(Xk_next' * Ai{i} * Xk_next); 
end

% Convex term + non-convex (g0 = 0) + regularizer
objective_prev = max(obj_vector_prev) + trace(tau_k_prev'*s1_prev) + trace(tau_k_prev'*s2_prev);
objective_next = max(obj_vector_next) + trace(tau_k_next'*s1_next) + trace(tau_k_next'*s2_next);

% Default flag is not to stop the iterations
stop = false;

diff = norm(objective_prev - objective_next);

% Check if we actually want to stop iterating
if diff <= delta
    stop = true;
end

end
