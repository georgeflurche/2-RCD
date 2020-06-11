import sys
import os
sys.path.insert(1,'C:/Users/George/Desktop/2-RCD')
from datetime import datetime
import numpy as np
import random as rnd
import logging
import argparse
from coordDesc import coordDesc

logging.basicConfig(
    format='%(levelname)s [%(module)s]: %(message)s'
)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def np_array_to_string(arr, arr_name):
    n = len(arr)
    m = len(arr[0])
    content = arr_name + " = ["
    matrix = []
    for i in range(n):
        line = []
        for j in range(m):
            line.append(str(arr[i][j]))
        matrix.append(", ".join(line))
    content += ";\n ".join(matrix) + "];\n"
    return content

def matlab_save_inputs(out_dir, Q, q, a, b, l, u, n):
    OUT_FILE = out_dir+"/inputs.m"
    content = np_array_to_string(Q, "Q")
    content += np_array_to_string(q, "q")
    content += np_array_to_string(a, "a")
    content += "b = " + str(b) + ";\n"
    content += np_array_to_string(l, "l")
    content += np_array_to_string(u, "u")
    content += "n = " + str(n) + ";\n"

    with open(OUT_FILE, 'w') as f:
        f.write(content)

def generate_data(n):
    Q = np.random.randn(n, n)
    Q = np.matmul(Q, np.transpose(Q))
    q = np.random.randn(n, 1)
    a = np.random.randn(n, 1)
    b = 0
    u = np.random.rand(n, 1)
    l = -np.random.rand(n, 1)
    return Q, q, a, b, l, u

DEFAULT_EPSILON_VALUE = 0.001
DEFAULT_MAX_ITER_VALUE = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser('This script is used to validate coordDesc routine')
    parser.add_argument('--dimension', '-d', type=int, help='The dimension n')
    parser.add_argument('--load_from_dir', type=str, help=(
        'Path to the directory containing inputs.npy file with data already generated.'
        'If the dimension is specified, this argument will be ignored.'))
    parser.add_argument('--max_iter', type=int, default=None,
        help='Maximum number of iterations.')
    parser.add_argument('--eps',type=float, default=None, help='Error tolerance')
    args = parser.parse_args()

    if args.dimension:
        n = args.dimension
        eps = DEFAULT_EPSILON_VALUE if args.eps is None else args.eps
        maxIter = DEFAULT_MAX_ITER_VALUE if args.max_iter is None else args.max_iter
        Q, q, a, b, l, u = generate_data(n)
        _logger.info('Running coordDesc algorithm')
        cpu_time, full_iterations_num, x_opt, f_val = coordDesc(Q, q, a, b, l, u,
            eps, maxIter)
    
        # Create dump directory
        out_dir = "coordDesc_run_"+datetime.now().strftime("%Y_%m_%d_%H%M%S"+"_"+str(n))
        os.system('mkdir ' + out_dir)

        # Save the inputs and the result
        _logger.info('Saving python inputs')
        np.save(out_dir + '/inputs.npy', [Q, q, a, b, l, u, eps, maxIter])

        _logger.info('Saving python outputs')
        np.save(out_dir + '/outputs.npy', [cpu_time, full_iterations_num, x_opt, f_val])
    
        _logger.info('Saving matlab inputs')
        matlab_save_inputs(out_dir, Q, q, a, b, l, u, n)
    
    elif args.load_from_dir:
        # Check if the path specified exists
        if os.path.exists(args.load_from_dir+'/inputs.npy'):
            [Q, q, a, b, l, u, eps, maxIter] = np.load(args.load_from_dir+'/inputs.npy')
            eps = eps if args.eps is None else args.eps
            maxIter = maxIter if args.max_iter is None else args.max_iter
            _logger.info('Running coordDesc algorithm')
            cpu_time, full_iterations_num, x_opt, f_val = coordDesc(Q, q, a, b, l, u,
                eps, maxIter)
        else:
            _logger.error("The path "+args.load_from_dir+'/inputs.npy'+
            " given as input couldn't be found")
            sys.exit(1)
    else:
        _logger.error("At least on of 'dimension' or 'load_from_dir' arguments must "
            "be specified!")
        _logger.info("Exiting validation script.")
        sys.exit(1)        
               
    _logger.info('The computed optimal solution is:\n'+str(x_opt))
    _logger.info('Full iteration number: '+str(full_iterations_num))
    _logger.info('Minimum value found:'+str(f_val))
    _logger.info('CPU time: '+str(cpu_time))