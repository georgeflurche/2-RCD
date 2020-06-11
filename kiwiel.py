"""
This script contains the method kiwiel, which solves Knapsack problem.
This code corresponds to PEP8 standard's requirements.
"""
import numpy as np
from math import ceil
import logging
from random import random
from functools import reduce

logging.basicConfig(
    format='%(levelname)s [%(module)s]: %(message)s'
)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

'''
Knapsack problem:
    min 0.5 x'Dx-a'x
    s.t.: b'x=r, l<=x<=u
Matrix D=diag(d), with d>0 (convex objective function)
K.C. Kiwiel,
On Linear-Time Algorithms for the Continuous Quadratic Knapsack Problem,
J. Optim Theory Appl (2007) 134: 549–554.
where:
    n - scalar: dimension
    r - scalar
    D - array of size nx1
    a - array of size nx1
    b - array of size nx1
    u - array of size nx1 (upper limit)
    l - array of size nx1 (lower limit)
'''


def kiwiel(n, r, D, a, b, u, l):
    dim = 2 * n
    Ind = [i for i, b_val in enumerate(b) if b_val < 0]
    lold = list(l)
    a = np.asarray([b_sig * a_val for a_val, b_sig in zip(a, np.sign(b))])
    b = np.asarray([b_sig * b_val for b_val, b_sig in zip(b, np.sign(b))])

    for i in Ind:
        l[i] = -u[i]
        u[i] = -lold[i]

    tl = []
    tu = []

    for i in range(n):
        tl_val = (a[i][0] - l[i][0] * D[i][0]) / b[i][0]
        tu_val = (a[i][0] - u[i][0] * D[i][0]) / b[i][0]

        tl.append(tl_val)
        tu.append(tu_val)

    tL = min(tu)
    tU = max(tl)
    T = tl + tu

    while(dim > 0):
        ran = int(random() * dim)
        tmed1 = T[ran]

        gt = gfunc(n, D, a, b, u, l, tmed1, tL, tU)

        if gt == r:
            tstar = tmed1
            _logger.info("Break was called")
            break
        elif gt > r:
            tL = tmed1
            T = [t_it for t_it in T if t_it > tmed1]
            dim = len(T)
        else:
            tU = tmed1
            T = [t_it for t_it in T if t_it < tmed1]
            dim = len(T)

        p = 0
        q = 0
        s = 0

        for i in range(n):
            if tl[i] <= tL:
                s = s + b[i][0] * l[i][0]

            if tu[i] >= tU:
                s = s + b[i][0] * u[i][0]

            if (tu[i] <= tL) and (tL <= tl[i]) and (tu[i] <= tU) \
                    and (tU <= tl[i]):
                p = p + a[i][0] * b[i][0] / D[i][0]
                q = q + b[i][0] * b[i][0] / D[i][0]

        if p == s == r == q == 0:
            tstar = None
        else:
            tstar = (p + s - r) / q

    xtstar = np.zeros((n, 1))
    for i in range(n):
        if tstar <= tu[i]:
            xtstar[i][0] = u[i][0]
        elif tu[i] <= tstar and tstar <= tl[i]:
            xtstar[i][0] = (a[i][0] - tstar * b[i][0]) / D[i][0]
        elif tl[i] <= tstar:
            xtstar[i][0] = l[i][0]

    for i in Ind:
        xtstar[i][0] = -xtstar[i][0]
    return xtstar


def gfunc(n, D, a, b, u, l, t, tL, tU):
    gt = 0

    for i in range(n):
        tli = (a[i][0] - l[i][0] * D[i][0]) / b[i][0]
        tui = (a[i][0] - u[i][0] * D[i][0]) / b[i][0]
        if t <= tui:
            gt = gt + b[i][0] * u[i][0]
        elif tui <= t and t <= tli:
            gt = gt + b[i][0] * (a[i][0] - t * b[i][0]) / D[i][0]
        elif tli <= t:
            gt = gt + b[i][0] * l[i][0]
    return gt
