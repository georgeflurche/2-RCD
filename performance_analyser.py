"""
This script is used to compare the SVM classifications with the 2-RCD
classifications. It gets as inputs the same bunch of data and runs both
algorithms, computing their accuracies and CPU times.
This code corresponds to PEP8 standard's requirements.
"""
import numpy as np
import sys
import os
import time
import logging
import argparse
import json
import psutil
from scipy.sparse import lil_matrix
sys.path.insert(0, 'libsvm/libsvm-3.24/python')
from svm import *                               # nopep8
from svmutil import *                           # nopep8
from commonutil import *                        # nopep8
from Adaptor import Adaptor                     # nopep8
from coordDesc import coordDesc                 # nopep8

logging.basicConfig(
    format='%(levelname)s [%(module)s]: %(message)s'
)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def count_zeros(K, sparse_matrix, threshold):
    Q = K.toarray() if sparse_matrix else K
    unique, counts = np.unique(Q, return_counts=True)
    result = dict(zip(unique, counts))
    zero_occurrence_percentage = result.get(0, 0)/len(Q)**2 * 100
    if zero_occurrence_percentage < threshold and sparse_matrix:
        _logger.warning('0 element occurrence percentage in Kernel matrix is '
                        + str(round(zero_occurrence_percentage, 4))+'%. '
                        'Setting "sparse_matrix" to false will be more '
                        'efficient')
    elif zero_occurrence_percentage >= threshold and not sparse_matrix:
        _logger.warning('0 element occurrence percentage in Kernel matrix is '
                        + str(round(zero_occurrence_percentage, 4))+'%. '
                        'Setting "sparse_matrix" to true will consume '
                        'less memory')


def check_generation_memory(data_size, number_of_features, sparse_matrix):
    total_memory = psutil.virtual_memory().total
    necessary_memory = data_size**2 * 64 / 8
    available_memory = psutil.virtual_memory().available
    if not sparse_matrix:
        _logger.info('Memory required to store the Kernel: ' +
                     str(round(necessary_memory/2**30, 2))+'GB')
    if necessary_memory > total_memory and not sparse_matrix:
        _logger.error("The Kernel matrix can't be generated as a dense matrix."
                      " Set the parameter 'sparse_matrix' to true if the "
                      "matrix is expected to be sparse.")
        sys.exit(1)
    elif necessary_memory > available_memory and not sparse_matrix:
        _logger.warning("The Kernel matrix might not be generated as a dense "
                        "matrix. Set the parameter 'sparse_matrix' to true "
                        "if the matrix is expected to be sparse")


def extract_features_from_data(x_train, x_test):
    _logger.info('Extracting the number of features')
    train_features = max([max(i.keys()) for i in x_train])
    test_features = max([max(i.keys()) for i in x_test])
    return max(train_features, test_features)


def K_poly(x, x1, coef0, gamma, degree):
    return (coef0 + gamma * np.transpose(x).dot(x1))**degree


def K_radial(x, x1, gamma):
    result = np.exp(-gamma*np.linalg.norm(x-x1, ord=2, axis=1))

    return np.array(result)


def get_classificator(kernel_type, x_opt, npx_train, npy_train, npx_test,
                      gamma, coef0, degree):
    if kernel_type == "linear":
        w = np.zeros((1, len(npx_train[0])))[0]
        _logger.info("Computing plane components (w, d)")
        for i in range(len(npy_train)):
            w += x_opt[i][0]*npy_train[i][0]*npx_train[i]
        for i, elem in enumerate(x_opt):
            x_i = elem[0]
            if x_i != 0:
                d = w.dot(np.transpose(npx_train[i])) - 1/npy_train[i][0]
                break
        classificator = np.matmul(w, np.transpose(npx_test))-d
    elif kernel_type == 'polynomial':
        _logger.info("Computing classificator's components for polynomial "
                     "kernel")
        for i, elem in enumerate(x_opt):
            x_i = elem[0]
            if x_i != 0:
                s = 0
                for j in range(len(npy_train)):
                    s += x_opt[j][0]*npy_train[j][0]*K_poly(
                        npx_train[j], npx_train[i], coef0, gamma, degree)
                d = s - 1/npy_train[i][0]
                break
        s1 = np.zeros((1, len(npx_test)))[0]
        for i, elem in enumerate(x_opt):
            xopt_i = elem[0]
            s1 += xopt_i*npy_train[i][0]*K_poly(
                np.reshape(npx_train[i], (len(npx_train[0]), 1)),
                np.transpose(npx_test), coef0, gamma, degree)[0]
        classificator = s1 - d
    else:
        _logger.info("Computing classificator's components for radial basis "
                     "kernel")
        for i, elem in enumerate(x_opt):
            x_i = elem[0]
            if x_i != 0:
                s = 0
                for j in range(len(npy_train)):
                    s += x_opt[j][0]*npy_train[j][0]*K_radial(
                        npx_train[j], np.array([npx_train[i]]), gamma)
                d = s - 1/npy_train[i][0]
                break
        s1 = np.zeros((1, len(npx_test)))[0]
        for i, elem in enumerate(x_opt):
            _logger.debug("Step "+str(i+1)+"/"+str(len(npy_train)))
            xopt_i = elem[0]
            s1 += xopt_i*npy_train[i][0]*K_radial(
                np.array([npx_train[i]]*len(npx_test)), npx_test, gamma)
        classificator = s1
    return classificator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to the config json file")
    parser.add_argument('--check_sparse', action='store_true',
                        help=("If this argument is set, a check will be made"
                              "on kernel matrix to count the number of zeros")
                        )
    args = parser.parse_args()

    if args.config_file:
        if not os.path.exists(args.config_file):
            _logger.error('The file '+args.config_file+' does not exist')
            sys.exit(1)
        else:
            with open(args.config_file) as f:
                config = json.load(f)

    IMPLEMENTED_KERNELS = {
        'linear': '0',
        'polynomial': '1',
        'radial_basis': '2'
        }
    maxiter = config.get('maxIter', 1000)
    train_data_path = config.get('train_data', '')
    test_data_path = config.get('test_data', '')
    epsilon = config.get('epsilon', 0.001)
    kernel_type = config.get('kernel_type', '')
    cost = config.get('cost', 1)
    degree = config.get('degree', 3)
    coef0 = config.get('coef0', 0)
    sparse_matrix = config.get('sparse_matrix', False)
    threshold = config.get('threshold', 50)

    y_train, x_train = svm_read_problem(train_data_path)
    y_test, x_test = svm_read_problem(test_data_path)
    data_size_train = len(y_train)
    data_size_test = len(y_test)
    features_num = extract_features_from_data(x_train, x_test)
    gamma = config.get('gamma', 1/features_num)

    adaptor_train = Adaptor(y=y_train, x=x_train, data_size=data_size_train,
                            features_num=features_num)
    adaptor_test = Adaptor(y=y_test, x=x_test, data_size=data_size_test,
                           features_num=features_num)
    npx_train = adaptor_train.adapt_x()
    npy_train = adaptor_train.adapt_y()
    npx_test = adaptor_test.adapt_x()
    npy_test = adaptor_test.adapt_y()

    lower_boundary = np.zeros((npy_train.shape[0], npy_train.shape[1]))
    upper_boundary = np.ones((npy_train.shape[0], npy_train.shape[1])) * cost
    q = np.ones((npy_train.shape[0], npy_train.shape[1]))

    check_generation_memory(data_size_train, features_num, sparse_matrix)

    if kernel_type == "linear":
        K = adaptor_train.linear_kernel(npx_train, npy_train, sparse_matrix)
    elif kernel_type == "polynomial":
        K = adaptor_train.polynomial_kernel(
            npx_train, npy_train, degree, coef0, gamma, sparse_matrix)
    elif kernel_type == "radial_basis":
        K = adaptor_train.radial_basis_kernel(npx_train, npy_train, gamma,
                                              sparse_matrix)
    else:
        _logger.error('The kernel_type '+kernel_type+' is not recognized')
        _logger.info("Kernels available: "+", ".join(
            IMPLEMENTED_KERNELS.keys()))
        sys.exit(1)
    if args.check_sparse:
        _logger.info('Counting the number of zeros in the Kernel matrix')
        count_zeros(K, sparse_matrix, threshold)

    _logger.info('Executing 2-RCD algorithm')
    cpu_time, full_iterations_num, x_opt, f_val = coordDesc(
        K, -q, npy_train, 0, lower_boundary, upper_boundary, epsilon, maxiter,
        sparse_matrix=sparse_matrix)

    _logger.info('2-RCD algorithm ended in '+str(cpu_time)+' seconds.')
    _logger.info('Full iteration number: '+str(full_iterations_num))
    _logger.info('Objectiv function/ Minimum found: '+str(f_val))

    classificator = get_classificator(kernel_type, x_opt, npx_train, npy_train,
                                      npx_test, gamma, coef0, degree)
    rcd_predicted_labels = [-1 if i < 0 else 1 for i in classificator]
    rcd_hits = int(sum(abs(np.array(rcd_predicted_labels) +
                   np.transpose(npy_test)[0])/2))
    rcd_accuracy = rcd_hits/data_size_test*100
    _logger.info('The accuracy of 2-RCD algorith for test data: ' +
                 str(round(rcd_accuracy, 4))+'% ('+str(rcd_hits)+'/' +
                 str(data_size_test)+')')

    _logger.info('Training the svm model')
    prob = svm_problem(np.transpose(npy_train).tolist()[0],
                       npx_train.tolist())
    start_time_svm = time.time()
    m = svm_train(prob, '-t '+IMPLEMENTED_KERNELS[kernel_type] +
                  ' -c '+str(cost)+' -d '+str(degree) +
                  ' -r '+str(coef0)+' -g '+str(gamma))
    end_time_svm = time.time()
    _logger.info('SVM training ended in '+str(end_time_svm - start_time_svm) +
                 ' seconds')
    _logger.info('Performing classification using SVM over the test data')
    svm_predicted_labels, _, _ = svm_predict(y_test, x_test, m)
