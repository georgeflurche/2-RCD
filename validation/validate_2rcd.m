clear;
script_name = 'coordDesc_run_2020_06_10_182633_500/inputs.m';
run(script_name)

eps = 10^-10;
maxIter = 5000000;
m = 10;
[cpu_time, full_iterations_num, x_opt, f_val] = coordDesc(Q, q, a, b, l, u, eps, maxIter, m);

% Construct and solve the model
fprintf(1,'Computing the optimal solution ...');
cvx_begin
    variable x(n)
    minimize ( (1/2)*quad_form(x,Q) + q'*x)
    subject to 
        a'*x == b;
        x >= l;
        x <= u;
cvx_end
fprintf(1,'Done! \n');

% Display results
disp('------------------------------------------------------------------------');
disp('The computed optimal solution is: ');
% disp(x);

