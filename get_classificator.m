function [classificator] = get_classificator(kernel_type, x_opt, ...
    x_train, y_train, x_test, gamma, coef0, degree)
    [n, m] = size(x_train);
    n_test = length(x_test);
    if strcmp(kernel_type, 'linear')
        w = zeros(1, m);
        for i = 1:n
            w = w + x_opt(i)*y_train(i)*x_train(i, :);
        end
        for i = 1:n
            if x_opt(i) ~= 0
                d = w*x_train(i, :)' - 1/y_train(i);
                break;
            end
        end
        classificator = w*x_test' - d;
    elseif strcmp(kernel_type, 'polynomial')
        for i = 1:n
            if x_opt(i) ~= 0
                s = 0;
                for j = 1:n
                    ki = K_poly(x_train(j, :), x_train(i, :), coef0, ...
                                gamma, degree);

                    s = s + x_opt(j)*y_train(j)*ki;
                end
                d = s - 1/y_train(i);
                break;
            end
        end
        s1 = zeros(1, n_test);
        for i = 1:n
            ki = K_poly(x_train(i, :), x_test, coef0, gamma, degree);
            s1 = s1 + x_opt(i)*y_train(i)* ki;
        end
        classificator = s1 - d;
    elseif strcmp(kernel_type, 'radial_basis')
        for i = 1:n
            if x_opt(i) ~= 0
                s = 0;
                for j = 1:n
                    ki = K_radial(x_train(j, :), x_train(i, :), gamma);
                    s = s + x_opt(j)*y_train(j)*ki;
                end
                d = s - 1/y_train(i);
                break;
            end
        end
        s1 = zeros(1, n_test);
        for i = 1:n
            ki = K_radial(repmat(x_train(i, :), n_test, 1), x_test, gamma);
            s1 = s1 + x_opt(i)*y_train(i)* ki';
        end
        classificator = s1 - d;
    end
end

function [kp] = K_poly(x, x1, coef0, gamma, degree)
    kp = (coef0 + gamma * (x*x1')).^degree;
end

function [kr] = K_radial(x, x1, gamma)
    aux = x-x1;
    kr = exp(-gamma*sqrt(sum(aux.^2,2)));
end