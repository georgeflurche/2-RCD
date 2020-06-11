import sys
import os
sys.path.insert(1,'C:/Users/George/Desktop/2-RCD')
from datetime import datetime
import numpy as np
import argparse
import logging
from kiwiel import kiwiel

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

def matlab_save_inputs(out_dir, n, r, d, a, b, u, l):
    OUT_FILE = out_dir+"/inputs.m"
    content = "n = " + str(n) + ";\n"
    content += "r = " + str(r) + ";\n"
    content += np_array_to_string(d, "d")
    content += np_array_to_string(a, "a")
    content += np_array_to_string(b, "b")
    content += np_array_to_string(l, "l")
    content += np_array_to_string(u, "u")

    with open(OUT_FILE, 'w') as f:
        f.write(content)

def generate_data(n):
    _logger.info('Generating data using dimension '+str(n))
    r = 0
    d = np.random.rand(n,1)
    a = 10*np.random.randn(n, 1)
    b = np.random.randn(n, 1)
    u = 10*np.random.rand(n, 1)
    l = np.zeros((n, 1))
    return r, d, a, b, u, l

if __name__ == "__main__":
    parser = argparse.ArgumentParser('This script is used to validate kiwiel routine')
    parser.add_argument('--dimension', '-d', type=int, help='The dimension n')
    parser.add_argument('--load_from_dir', type=str, help=(
        'Path to the directory containing inputs.npy file with data already generated.'
        'If the dimension is specified, this argument will be ignored.'))
    args = parser.parse_args()

    if args.dimension:
        n = args.dimension
        r, d, a, b, u, l = generate_data(n)
        _logger.info('Running kiwiel algorithm')
        xstar = kiwiel(n, r, d, a, b, u, l)
        
        # Create dump directory
        out_dir = "kiwiel_run_"+datetime.now().strftime("%Y_%m_%d_%H%M%S"+"_"+str(n))
        os.system('mkdir ' + out_dir)

        # Save the inputs and the result
        _logger.info('Saving python inputs')
        np.save(out_dir + '/inputs.npy', [n, r, d, a, b, u, l])
        
        _logger.info('Saving python outputs')
        np.save(out_dir + '/outputs.npy', xstar)
        
        _logger.info('[INFO] Saving matlab inputs')
        matlab_save_inputs(out_dir, n, r, d, a, b, u, l)

    elif args.load_from_dir:
        # Check if the path specified exists
        if os.path.exists(args.load_from_dir+'/inputs.npy'):
            [n, r, d, a, b, u, l] = np.load(args.load_from_dir+'/inputs.npy')
            _logger.info('Running kiwiel algorithm')
            xstar = kiwiel(n, r, d, a, b, u, l)
        else:
            _logger.error("The path "+args.load_from_dir+'/inputs.npy'+
            " given as input couldn't be found")
            sys.exit(1)
    else:
        _logger.error("At least one argument must be specified!")
        _logger.info("Exiting validation script.")
        sys.exit(1)
   
    _logger.info('The computed optimal solution is :')
    print(xstar)


