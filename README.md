Authors: Ion Necoară, George Flurche
The 2-RCD code is able to solve huge structured convex  quadratic programs with a single linear equality constraints and box constraints  using the algorithm 2-RCD developed in the paper: I. Necoara, A. Patrascu, A random coordinate descent algorithm for optimization problems with composite objective function and linear coupled constraints, Computational Optim. & Applications, 57(2): 307-337, 2014. 
2-RCD algorithm solves the following problem:
    =================================================
    min (0.5*transpose(x)*Q*x + transpose(q)*x), with:
    transpose(a)*x=b and x is in range [l, u]
    =================================================
The algorithm is implemented in both Python and Matlab programming languages and it was validated by a comparison with the modeling system for convex optimization- cvx
Hence, to analyse the performance of the algorithm, we trained it with data provided by libsvm and performed the same classification using both 2-RCD and svm and compared the results
