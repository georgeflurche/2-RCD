"""
This script contains the method coordDesc, which executes 2-RCD altorithm.
This code corresponds to PEP8 standard's requirements.
"""
import numpy as np
import random as rnd
from scipy.sparse import lil_matrix
import time
import logging
from kiwiel import kiwiel

logging.basicConfig(
    format='%(levelname)s [%(module)s]: %(message)s'
)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def update_f_val_list(f_val_list, f_val_new, m):
    if len(f_val_list) < m:
        f_val_list.append(f_val_new)
    else:
        f_val_list = f_val_list[1:] + [f_val_new]
    return f_val_list


def check_absolute_oscilation(eps, f_val_list, m):
    if len(f_val_list) < m:
        return False
    else:
        for i in range(m-1):
            absolute_oscilation = abs(f_val_list[i+1] - f_val_list[i])
            if absolute_oscilation > eps:
                return False
    return True


def coordDesc(Q, q, a, b, l, u, eps, maxIter, m=10, sparse_matrix=False):
    '''
    This function performs 2-RCD algorithm and tries to solve the
    following problem:
    =================================================
    min (0.5*transpose(x)*Q*x + transpose(q)*x), with:
    transpose(a)*x=b and x is in range [l, u]
    =================================================
    INPUT:
    - Q is a quadratic matrix in R(NxN)
    - q is an array in R(N)
    - a is an array in R(N)
    - b is a scalar in R
    - l is an array in R(N), representing the lower bound
    - u is an array in R(N), representing the upper bound
    - eps represent the accuracy of the solution
    - maxIter is the maximum number of iteration which the algorithm
      is allowed to perform.
    - m is the number of consecutive objective functions stored in f_val_list;
      if the absolute difference between every two consecutive elements in the
      list is smaller than eps, the algorithm will stop
    - sparse_matrix is a boolean variable which enables/disables the sparse
      format of the matrices. For more details about sparse matrices check
      https://docs.scipy.org/doc/scipy/reference/sparse.html

    OUTPUT:
    - cpu_time, representing the time in seconds needed for the
      function to solve the problem
    - full_iterations_num
    - x_opt, representing the solution x of the problem
    - f_val, representing the minimum value of the function found
    '''
    # ============== Initialization ===============

    N = Q.get_shape()[0] if sparse_matrix else len(Q)
    x = kiwiel(N, b, np.ones((N, 1)), np.zeros((N, 1)), a, u, l)
    x = lil_matrix(x) if sparse_matrix else x
    x_trans = x.transpose() if sparse_matrix else np.transpose(x)
    q_trans = lil_matrix(q).transpose() if sparse_matrix else np.transpose(q)
    f_val = 0.5 * x_trans.dot(Q).dot(x) + q_trans.dot(x)
    f_val_list = [f_val[0, 0]] if sparse_matrix else [f_val[0][0]]
    startT = time.time()

    # ==============   Loop steps  ===============
    for iter in range(maxIter):
        if check_absolute_oscilation(eps, f_val_list, m):
            _logger.info(
                'The absolute oscilation is smaller or equal than epsilon '
                'for the last '+str(m)+' iterations. ')
            _logger.info(
                '2-RCD was interrupted after '+str(iter)+' iterations')
            break
        # Select two different random indexes in range [0, N-1]
        i = rnd.randint(0, N-1)
        j = rnd.choice(list(set(range(0, N))-{i}))

        Q_i_row = Q.getrow(i).toarray()[0] if sparse_matrix else Q[i][:]
        Q_j_row = Q.getrow(j).toarray()[0] if sparse_matrix else Q[j][:]

        Qii = Q_i_row[i] if sparse_matrix else Q_i_row[i]
        Qjj = Q_j_row[j] if sparse_matrix else Q_j_row[j]
        Qij = Q_i_row[j] if sparse_matrix else Q_i_row[j]

        Lij = 0.5*(Qii + Qjj + ((Qii-Qjj)**2 + 4*Qij**2)**0.5)

        Qi_dot_x = (
            Q_i_row.dot(x.toarray()) if sparse_matrix else Q_i_row.dot(x))
        Qj_dot_x = (
            Q_j_row.dot(x.toarray()) if sparse_matrix else Q_j_row.dot(x))

        grad_i = (Qi_dot_x + q[i][0])[0]
        grad_j = (Qj_dot_x + q[j][0])[0]

        '''
        For each iteration, solving the sub-problem:
            min (0.5*Lij*(si^2 + sj^2) + grad_i*si +grad_j*sj)
                subject to:
                    ai*si + aj*sj = 0
                    xi + si in range [li, ui]
                    xj + sj in range [lj, uj]
        '''
        # Notations:
        ai = a[i][0]
        aj = a[j][0]
        xi = x[i, 0] if sparse_matrix else x[i][0]
        xj = x[j, 0] if sparse_matrix else x[j][0]
        li = l[i][0]
        lj = l[j][0]
        ui = u[i][0]
        uj = u[j][0]

        if ai != 0:
            # Case I
            if -aj/ai > 0:
                # Sub-case I.1
                left = max(-(li-xi)*ai/aj, lj-xj)
                right = min(-(ui-xi)*ai/aj, uj-xj)
                sol = -(grad_j - grad_i*aj/ai)/(Lij*(aj**2/ai**2 + 1))
                sj_opt = min(max(left, sol), right)
                si_opt = -aj/ai*sj_opt
            elif -aj/ai < 0:
                # Sub-case I.2
                left = max(-(ui-xi)*ai/aj, lj-xj)
                right = min(-(li-xi)*ai/aj, uj-xj)
                sol = -(grad_j - grad_i*aj/ai)/(Lij*(aj**2/ai**2 + 1))
                sj_opt = min(max(left, sol), right)
                si_opt = -aj/ai*sj_opt
            else:
                # Sub-case I.3
                si_opt = 0
                sj_opt = min(max(lj-xj, -grad_i/Lij), uj-xj)
        else:
            # Case II
            if aj == 0:
                # Sub-case II.1
                si_opt = min(max(li-xi, -grad_i/Lij), ui-xi)
                sj_opt = min(max(lj-xi, -grad_j/Lij), uj-xj)
            else:
                # Sub-case II.2
                sj_opt = 0
                si_opt = min(max(li-xi, -grad_i/Lij), ui-xi)

        # Update the array x
        if sparse_matrix:
            x[i, 0] = xi + si_opt
            x[j, 0] = xj + sj_opt
            x_trans = x.transpose()
        else:
            x[i][0] = xi + si_opt
            x[j][0] = xj + sj_opt
            x_trans = np.transpose(x)

        # Calculate the objective function
        f_val_new = 0.5 * x_trans.dot(Q).dot(x) + q_trans.dot(x)
        f_val = f_val_new[0, 0] if sparse_matrix else f_val_new[0][0]
        f_val_list = update_f_val_list(f_val_list, f_val_new, m)
    x_opt = x.toarray() if sparse_matrix else x
    endT = time.time()
    cpu_time = endT - startT
    full_iterations_num = int(2*iter/N)

    return cpu_time, full_iterations_num, x_opt, f_val
