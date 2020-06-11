function [maxiter, train_data_path, test_data_path, epsilon, ...
          kernel_type, cost, degree, coef0, sparse_matrix, gamma] = ...
          configuration_reader(table_name)
    maxiter = 1000;
    train_data_path = '';
    test_data_path = '';
    epsilon = 0.001;
    kernel_type = '';
    cost = 1;
    degree = 3;
    coef0 = 1;
    sparse_matrix = 0;
    gamma = 1/10;

    T = readtable(table_name);
    config = table2array(T);
    c_length = length(config);

    for i = 1:c_length
        if strcmp(config(i, 1), 'maxIter')
            maxiter = str2double(config(i, 2));
        elseif strcmp(config(i, 1), 'train_data')
            train_data_path = char(config(i, 2));
        elseif strcmp(config(i, 1), 'test_data')
            test_data_path = char(config(i, 2));
        elseif strcmp(config(i, 1), 'epsilon')
            epsilon = str2double(config(i, 2));
        elseif strcmp(config(i, 1), 'kernel_type')
            kernel_type = char(config(i, 2));
        elseif strcmp(config(i, 1), 'cost')
            cost = str2double(config(i, 2));
        elseif strcmp(config(i, 1), 'degree')
            degree = str2double(config(i, 2));
        elseif strcmp(config(i, 1), 'coef0')
            coef0 = str2double(config(i, 2));
        elseif strcmp(config(i, 1), 'sparse_matrix')
            if strcmp(config(i, 2), 'TRUE')
                sparse_matrix = 1;
            else
                sparse_matrix = 0;
            end
        elseif strcmp(config(i, 1), 'gamma')
            gamma = str2double(config(i, 2));
        end
    end
end