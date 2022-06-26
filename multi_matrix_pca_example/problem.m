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
clear all; close all; clc;
% Problem initial values
m = 10;             % number of lines of X
n = 100;            % number of cols of X
p = 8;              % number of A matrices
tau_0 = 0.5;
mu = 1.05;
tau_max = 10000;
delta = 1E-4;       % Stoping constant

number_of_experiments = 1000;

number_steps_per_run = zeros(number_of_experiments,1);
avg_time_per_run = zeros(number_of_experiments,1);
objective_val_per_run = zeros(number_of_experiments,1);

% Perform the experiments in parallel for the average initialization
parfor i=1:number_of_experiments
   % Perform PCA
   [number_steps_per_run(i), avg_time_per_run(i), objective_val_per_run(i)] = perform_optimization(m, n, p, tau_0, mu, tau_max, delta, 0);
end

save('results_avg_initialization.mat', 'number_steps_per_run', 'avg_time_per_run', 'objective_val_per_run')

% Perform the experiments in parallel for the best initialization+
parfor i=1:number_of_experiments
   % Perform PCA
   [number_steps_per_run(i), avg_time_per_run(i), objective_val_per_run(i)] = perform_optimization(m, n, p, tau_0, mu, tau_max, delta, 1);
end

save('results_best_initialization.mat', 'number_steps_per_run', 'avg_time_per_run', 'objective_val_per_run')

%% Plot the results for the average initialization
load('results_avg_initialization.mat')
hist(objective_val_per_run, [10:0.05:12.5], 10);
xlabel('objective value');
ylabel('number of trials');

%% Plot the results for the best initialization
load('results_best_initialization.mat')
hist(objective_val_per_run, [10:0.05:12.5], 10);
xlabel('objective value');
ylabel('number of trials');
