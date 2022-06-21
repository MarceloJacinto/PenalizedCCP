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