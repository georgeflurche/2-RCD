import sys
import numpy as np
from scipy.sparse import lil_matrix
import logging

logging.basicConfig(
    format='%(levelname)s [%(module)s]: %(message)s'
)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class Adaptor:
    '''
    This class is used to convert data from svmlib format into
    numpy standard matrix/arrays
    '''
    def __init__(self, y, x, data_size, features_num, classes=2):
        self.x = x
        self.y = y
        self.data_size = data_size
        self.features_num = features_num
        self.classes = classes

    def adapt_x(self):
        """
        Method that adapts x into a numpy matrix Nxm, where:
        - N is the number of inputs (data)
        - m is the number of features
        """
        new_x = []
        for data in self.x:
            line = []
            for i in range(1, self.features_num+1):
                line.append(data.get(i, 0))
            new_x.append(line)
        return np.array(new_x)

    def adapt_y(self):
        """
        Method that adapts y into a numpy array Nx1, where:
        -N is the number of inputs
        """
        new_y = np.array(self.y)
        new_y.shape = (self.data_size, 1)
        return new_y

    def linear_kernel(self, x, y, sparse_matrix):
        """
        Method that generates linear Kernel matrix of size NxN
        """
        _logger.info('Generating linear Kernel matrix')
        if sparse_matrix:
            x = lil_matrix(x)
            diag_y = lil_matrix(np.diagflat(y))
            xxT = x.dot(x.transpose())
            dyxxT = diag_y.dot(xxT)
            K = dyxxT.dot(diag_y)
        else:
            K = np.diagflat(y).dot(x.dot(np.transpose(x))).dot(np.diagflat(y))
        return K

    def polynomial_kernel(self, x, y, degree, coef0, gamma, sparse_matrix):
        """
        Method that generates polynomial Kernel matrix of size NxN
        """
        _logger.info('Generationg polynomial Kernel matrix')
        if sparse_matrix:
            _logger.error('Sparse polynomial kernel is not supported yet')
            sys.exit(1)
        else:
            K = np.diagflat(y).dot((gamma*x.dot(
                np.transpose(x)) + coef0)**degree).dot(np.diagflat(y))
        return K

    def radial_basis_kernel(self, x, y, gamma, sparse_matrix):
        """
        Method that generates polynomial Kernel matrix of size NxN
        using the formula K(ai,aj) = e^(-gamma*||ai-aj||)
        """
        _logger.info('Generating radial basis Kernel matrix')
        if sparse_matrix:
            _logger.error('Sparse radial basis kernel is not supported yet')
            sys.exit(1)
        else:
            N = len(x)
            norm_matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    norm_matrix[i][j] = np.linalg.norm(x[i] - x[j], ord=2)
            Kx = np.exp(-gamma*norm_matrix)
        return np.matmul(np.matmul(np.diagflat(y), Kx), np.diagflat(y))
