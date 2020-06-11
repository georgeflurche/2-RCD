function [cpu_time, full_iterations_num, x_opt, f_val] = coordDesc(Q, q, a, b, l, u, eps, maxIter, m)
%     This function performs 2-RCD algorithm and tries to solve the
%     following problem:
%     =================================================
%     min (0.5*transpose(x)*Q*x + transpose(q)*x), with:
%     transpose(a)*x=b and x is in range [l, u]
%     =================================================
%     INPUT:
%     - Q is a quadratic matrix in R(NxN)
%     - q is an array in R(N)
%     - a is an array in R(N)
%     - b is a scalar in R
%     - l is an array in R(N), representing the lower bound
%     - u is an array in R(N), representing the upper bound
%     - eps represent the accuracy of the solution
%     - maxIter is the maximum number of iteration which the algorithm
%       is allowed to perform.
%     - m is the number of consecutive objective functions stored in f_val_list;
%       if the absolute difference between every two consecutive elements in the
%       list is smaller than eps, the algorithm will stop
% 
%     OUTPUT:
%     - cpu_time, representing the time in seconds needed for the
%       function to solve the problem
%     - full_iterations_num
%     - x_opt, representing the solution x of the problem
%     - f_val, representing the minimum value of the function found

    n = length(Q);
    x = kiwiel(n, b, ones(n, 1), zeros(n, 1), a, u, l);
%   f_val represents the value of the objective function with the current x
    f_val = 0.5*x'*Q*x + q'*x;
    
%   Populate f_val_list with arbitrary big values (>> eps)
%   This trick is used to avoid changing size of f_val_list on the fly
%   f_val_list is used to store the values of the objective functions of
%   the last m iteration
    f_val_list = [randi([1, 20], 1, m-1)+f_val, f_val];
    iter = 0;
    
    tStart = tic;
    
%   The while loop keeps running as long as absolute oscilation is over
%   eps and the current iteration is smaller than maxIter
    while (check_absolute_oscilation(eps, f_val_list, m) && iter < maxIter)
%       Generate two different values i and j in the range [1, n]
        i = randi([1, n], 1, 1);
        j = randi([1, n], 1, 1);
        while (i == j)
            % regenerate j until it's different from i
            j = randi([1, n], 1, 1);
        end
        
        Lij = 0.5*(Q(i,i)+Q(j,j)+((Q(i,i)-Q(j,j))^2+4*Q(i,j)^2)^0.5);
        grad_i = Q(i,:)*x + q(i);
        grad_j = Q(j,:)*x + q(j);
        
        if (a(i) ~= 0)
            % case I  
            if(-a(j)/a(i) > 0)
                % subcase I.1
                left = max(-(l(i)-x(i))*a(i)/a(j), l(j)-x(j));
                right = min(-(u(i)-x(i))*a(i)/a(j), u(j)-x(j));
                sol = -(grad_j - grad_i*a(j)/a(i))/(Lij*(a(j)^2/a(i)^2+1));
                sj_opt = min(max(left, sol), right);
                si_opt = -a(j)/a(i)*sj_opt;
            end
            if (-a(j)/a(i) < 0)
                % subcase I.2
                left = max(-(u(i)-x(i))*a(i)/a(j), l(j)- x(j));
                right = min(-(l(i)-x(i))*a(i)/a(j), u(j)- x(j));
                sol = -(grad_j - grad_i*a(j)/a(i))/(Lij*(a(j)^2/a(i)^2+1));
                sj_opt = min(max(left, sol), right);
                si_opt = -a(j)/a(i)*sj_opt;
            end
            if (-a(j)/a(i) == 0)
                % subcase I.3
                si_opt = 0;
                sj_opt = min(max(l(j)-x(j),-grad_j/Lij),u(j)-x(j));
            end
        end
        if( a(i)== 0)
            % case II
            if(a(j) == 0)
                % subcase II.1
                si_opt = min(max(l(i)-x(i),-grad_i/Lij),u(i)-x(i));
                sj_opt = min(max(l(j)-x(j),-grad_j/Lij),u(j)-x(j));
            end
            if(a(j) ~= 0)
                % subcase II.2
                sj_opt = 0;
                si_opt = min(max(l(i)-x(i),-grad_i/Lij),u(i)-x(i));
            end
               
        end
        
        x(i) = x(i) + si_opt;
        x(j) = x(j) + sj_opt;
        f_val_new = 0.5*x'*Q*x + q'*x;
        f_val_list = [f_val_list(2:m) f_val_new];
        f_val = f_val_new;
        iter = iter + 1;
    end
    cpu_time = toc(tStart);
    full_iterations_num = fix(2*iter/n);
    x_opt = x;
end

function [condition] = check_absolute_oscilation(eps, f_val_list, m)
%     This function is checking the difference between the elements of
%     f_val_list. If at least one pair of consecutive elements has the 
%     absolute difference bigger than eps, the function will return 1
%     and the while loop will continue to execute. If all the consecutive
%     pairs have the absolute difference smaller or equal with eps, the 
%     function will return 0 and the algorithm coordDesc will stop
  
    condition = 0;
    for i = 1:m-1
        absolute_oscilation = abs(f_val_list(i+1) - f_val_list(i));
        if (absolute_oscilation > eps)
            condition = 1;
            break
        end
    end
end