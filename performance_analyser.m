clear;
% Load the configuration file in the script
table_name = 'configuration_matlab/config_5_a2a_radial_basis.csv';
[maxiter, train_data_path, test_data_path, epsilon, kernel_type, cost, ...
 degree, coef0, sparse_matrix, gamma] = configuration_reader(table_name);

% Load training data
[y_train, x_train] = libsvmread(train_data_path);
[y_test, x_test] = libsvmread(test_data_path);

[data_size_train, features_num_train] = size(x_train);
[data_size_test, features_num_test] = size(x_test);

% Adjust the training set/ the testing set if it's the case
if features_num_train > features_num_test
    delta = features_num_train - features_num_test;
    z = zeros(data_size_test, delta);
    x_test = [x_test, z];
elseif features_num_test > features_num_train
    delta = features_num_test - features_num_train;
    z = zeros(data_size_train, delta);
    x_train = [x_train, z];
end

features_num = max(features_num_train, features_num_test);
gamma = 1/features_num;

% Create train parameters string for libsvm training method
if strcmp(kernel_type, 'linear')
    train_params = '-t 0';
    K = diag(y_train)*(x_train*x_train')*diag(y_train);
elseif strcmp(kernel_type, 'polynomial')
    train_params = '-t 1';
    K = diag(y_train)*((gamma*(x_train*x_train') + coef0).^degree) ...
        *diag(y_train);
elseif strcmp(kernel_type, 'radial_basis')
    train_params = '-t 2';
    norm_matrix = zeros(data_size_train);
    for i = 1:data_size_train
        for j = 1:data_size_train
            norm_matrix(i,j) = norm(x_train(i, :) - x_train(j, :));
        end
    end
    Kx = exp(-gamma * norm_matrix);
    K = diag(y_train)*Kx*diag(y_train);
else
    disp('Unrecognized kernel type');
    return;
end
train_params = strcat(train_params, ' -c', {' '}, int2str(cost), ' -d', ...
                      {' '}, int2str(degree), ' -r', {' '},  ...
                      int2str(coef0), ' -g ', {' '}, int2str(gamma));

% Running 2-RCD algorithm
lower_boundary = zeros(data_size_train, 1);
upper_boundary = ones(data_size_train, 1) * cost;
q = ones(data_size_train, 1);

[cpu_time_rcd, full_iterations_num, x_opt, f_val] = coordDesc(K, -q, ...
    y_train, 0, lower_boundary, upper_boundary, epsilon, maxiter, 10);

disp(strcat('2-RCD training ended in', {' '}, num2str(cpu_time_rcd), ...
            {' '}, 'seconds'));

% Calculating predictions
classificator = get_classificator(kernel_type, x_opt, x_train, y_train, ...
                                  x_test, gamma, coef0, degree);
rcd_predicted_labels = zeros(1, data_size_test);
for i = 1:data_size_test
    if classificator(i) < 0
        rcd_predicted_labels(i) = -1;
    else
        rcd_predicted_labels(i) = 1;
    end
end
rcd_hits = sum(abs(rcd_predicted_labels + y_test')/2);
rcd_accuracy = rcd_hits/data_size_test*100;
disp(strcat('2-RCD accuracy: ',{' '}, num2str(rcd_accuracy), '%')); 

% Training the svm model
svm_start_time = tic;
model = svmtrain(y_train(1:data_size_train,:), ...
                 x_train(1:data_size_train,:), train_params);

svm_end_time = toc(svm_start_time);
disp(strcat('SVM training ended in', {' '}, num2str(svm_end_time), ...
            {' '}, 'seconds'));
% Performing classification over the dataset using the trained model with
% svm
svmpredict(y_test, x_test, model);



