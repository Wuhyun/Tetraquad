''' tetraquad.py
Module for computing quadrature rules for numerical integration
over a "tetrapyd" volume in three dimensions,
which consists of points (k1, k2, k3) satisfying:
k_min <= k1, k2, k3 <= k_max
k1 + k2 >= k3
k2 + k3 >= k1
k3 + k1 >= k2
'''

import numpy as np
from scipy.special import gamma
from numpy.random import default_rng
import scipy
import cvxopt


def quad(func, k_min, k_max, N=10, grid=None, weights=None):
    ''' Integrate the given function over a tetrapyd volume.
    May provide a precomputed quadrature rule (grid, weights) to
    speed up the calculation.
    '''

    if grid is None or weights is None:
        grid, weights = quadrature(k_min, k_max, N)

    k1, k2, k3 = grid
    integrand = func(k1, k2, k3)
    eval = np.dot(weights, integrand)

    return eval


def quadrature(k_min, k_max, N):
    ''' Returns a quadrature rule that guarantees the integral of
    symmetric polynomials of order <= N over a tetrapyd is exact.
    '''

    ratio = k_min / k_max
    grid, weights = unit_quadrature_nnls(ratio, N)
    grid *= k_max
    weights *= ratio ** 3

    return grid, weights


def unit_quadrature_nnls(alpha, N, grid_type="Uniform"):
    ''' Returns a quadrature rule which has N grid points on each dimension.
    Minimises the integration error of symmetric polynomials 
    order <= f(N) over a tetrapyd specified by the triangle conditions and
    alpha <= k1, k2, k3 <= 1
    '''

    # Set up grid points
    if grid_type == "Uniform":
        i1, i2, i3 = uni_tetra_triplets(alpha, N)
        grid_1d = np.linspace(alpha, 1, N)
    elif grid_type == "GL":
        i1, i2, i3 = gl_tetra_triplets(alpha, N)
        grid_1d, _ = gl_quadrature(alpha, N)
    else:
        print("Grid name {} currently unsupported.".format(grid_type))

    num_weights = i1.shape[0]
    grid = np.zeros((3, num_weights))
    grid[0,:] = grid_1d[i1]
    grid[1,:] = grid_1d[i2]
    grid[2,:] = grid_1d[i3]

    # Prepare orthogonal polynomials
    M = N
    while True:
        # List of polynomial orders (p,q,r)
        ps, qs, rs = poly_triplets_total_degree(M)
        #ps, qs, rs = poly_triplets_individual_degree(M)
        num_polys = ps.shape[0]

        if num_polys > num_weights:
            M -= 1
            ps, qs, rs = poly_triplets_total_degree(M)
            #ps, qs, rs = poly_triplets_individual_degree(M)
            num_polys = ps.shape[0]
            break
        else:
            M += 1

    #M = 2 * N
    #ps, qs, rs = poly_triplets_total_degree(M)
    #num_polys = ps.shape[0]

    print("M =", M, ", N =", N)

    # Obtain orthonormalisation coefficients for the polynomials
    ortho_L = orthonormal_polynomials(ps, qs, rs, alpha)
    poly_ampl = np.copy(np.diag(ortho_L))
    ortho_L /= np.sqrt(poly_ampl)[:,np.newaxis]

    # Evaluations of the polynomials at grid points
    grid_evals = grid_poly_evaluations(grid_1d, ps, qs, rs, i1, i2, i3)
    grid_evals = np.matmul(ortho_L, grid_evals)

    # Base value for the weights
    tetra_volume = (1 - alpha ** 3) / 2.
    base_weights = np.ones(num_weights) * tetra_volume / num_weights
    
    # Analytic values for the integrals
    analytic = analytic_poly_integrals(ps, qs, rs, alpha)
    analytic = np.matmul(ortho_L, analytic)

    # We are now ready to compute the quadrature weights.
    # Non-Negative Least Squares:
    # minimise ||A x - b||^2 for non-negative x

    print(len(ps), len(base_weights))

    A = grid_evals
    #b = np.concatenate([[1], np.zeros(num_polys-1)])
    b = np.concatenate([[ortho_L[0,0]*tetra_volume], np.zeros(num_polys-1)])

    x, rnorm = scipy.optimize.nnls(A, b)

    weights = x

    return grid, weights



def unit_quadrature_qp(alpha, N, grid_type="Uniform"):
    ''' Returns a quadrature rule which has N grid points on each dimension.
    This should guarantee that the integral of
    symmetric polynomials of order <= f(N) over a tetrapyd is exact.
    Tetrapyd specified by the triangle conditions and
    alpha <= k1, k2, k3 <= 1
    '''

    # Set up grid points
    if grid_type == "Uniform":
        i1, i2, i3 = uni_tetra_triplets(alpha, N)
        grid_1d = np.linspace(alpha, 1, N)
    elif grid_type == "GL":
        i1, i2, i3 = gl_tetra_triplets(alpha, N)
        grid_1d, _ = gl_quadrature(alpha, N)
    else:
        print("Grid name {} currently unsupported.".format(grid_type))

    num_weights = i1.shape[0]
    grid = np.zeros((3, num_weights))
    grid[0,:] = grid_1d[i1]
    grid[1,:] = grid_1d[i2]
    grid[2,:] = grid_1d[i3]

    # Prepare orthogonal polynomials
    M = N // 2
    while True:
        # List of polynomial orders (p,q,r)
        ps, qs, rs = poly_triplets_total_degree(M)
        #ps, qs, rs = poly_triplets_individual_degree(M)
        num_polys = ps.shape[0]

        if num_polys > num_weights:
            M -= 1
            ps, qs, rs = poly_triplets_total_degree(M)
            #ps, qs, rs = poly_triplets_individual_degree(M)
            num_polys = ps.shape[0]
            break
        else:
            M += 1

    # Obtain orthonormalisation coefficients for the polynomials
    ortho_L = orthonormal_polynomials(ps, qs, rs, alpha)
    poly_ampl = np.copy(np.diag(ortho_L))
    ortho_L /= np.sqrt(poly_ampl)[:,np.newaxis]

    # Evaluations of the polynomials at grid points
    grid_evals = grid_poly_evaluations(grid_1d, ps, qs, rs, i1, i2, i3)
    grid_evals = np.matmul(ortho_L, grid_evals)

    # Base value for the weights
    tetra_volume = (1 - alpha ** 3) / 2.
    base_weights = np.ones(num_weights) * tetra_volume / num_weights
    
    # Analytic values for the integrals
    analytic = analytic_poly_integrals(ps, qs, rs, alpha)
    analytic = np.matmul(ortho_L, analytic)

    num_constraints = np.linalg.matrix_rank(grid_evals)


    # We are now ready to compute the quadrature weights.
    # Quadratic Programming:
    # minimise  (1/2) x^T P x + q^T x  subject to Gx <= h  and  Ax = b

    n_c = num_constraints
    print(len(ps), num_constraints, len(base_weights))

    #'''
    P = np.matmul(grid_evals[n_c:,:].T, grid_evals[n_c:,:])
    #P += np.max(np.abs(P)) * 1e-10 * np.identity(P.shape[0])
    qp_P = cvxopt.matrix(P)
    qp_q = cvxopt.matrix(-np.matmul(grid_evals[n_c:,:].T, analytic[n_c:]))
    qp_G = cvxopt.matrix(-np.identity(len(base_weights)))
    qp_h = cvxopt.matrix(np.zeros_like(base_weights))
    #qp_A = cvxopt.matrix(grid_evals[1:n_c,:])
    #qp_b = cvxopt.matrix(np.zeros(n_c-1))
    qp_A = cvxopt.matrix(grid_evals[:n_c,:])
    qp_b = cvxopt.matrix(np.concatenate([[1], np.zeros(n_c-1)]))
    qp_i = cvxopt.matrix(base_weights)
    #qp_i = cvxopt.matrix(np.ones_like(base_weights)/len(base_weights))

    qp_result = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b, initvals=qp_i)
    #qp_result = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)
    #qp_result = cvxopt.solvers.qp(qp_P, qp_q, A=qp_A, b=qp_b, initvals=qp_i)
    #'''

    '''
    qp_P = cvxopt.matrix(np.matmul(grid_evals[1:,:].T, grid_evals[1:,:]))
    qp_q = cvxopt.matrix(-np.matmul(grid_evals[1:,:].T, analytic[1:]))
    '''

    #qp_result = cvxopt.solvers.qp(qp_P, qp_q,initvals=qp_i)
    #qp_result = cvxopt.solvers.qp(qp_P, qp_q)

    weights = np.array(qp_result["x"])

    return grid, weights


def uniform_tetrapyd_weights(alpha, N, MC_N_SAMPLES=5000):
    ''' Returns a uniform tetrapyd grid with weights proportional to 
    the volume of the grid.
    '''

    # Set up grid points
    i1, i2, i3 = uni_tetra_triplets(alpha, N)
    grid_1d = np.linspace(alpha, 1, N)
    num_weights = i1.shape[0]

    grid = np.zeros((3, num_weights))
    grid[0,:] = grid_1d[i1]
    grid[1,:] = grid_1d[i2]
    grid[2,:] = grid_1d[i3]

    # 1D bounds and weights for k grid points
    interval_bounds = np.zeros(N + 1)
    interval_bounds[0] = grid_1d[0]
    interval_bounds[1:-1] = (grid_1d[:-1] + grid_1d[1:]) / 2
    interval_bounds[-1] = grid_1d[-1]
    k_weights = np.diff(interval_bounds)

    # Initialise weights based on symmetry and grid intervals
    tetrapyd_weights = k_weights[i1] * k_weights[i2] * k_weights[i3]
    tetrapyd_weights[(i1 != i2) & (i2 != i3)] *= 6   # Distinct indices
    tetrapyd_weights[(i1 != i2) & (i2 == i3)] *= 3   # Two identical indices
    tetrapyd_weights[(i1 == i2) & (i2 != i3)] *= 3   # Two identical indices

    # Further weights for points of the surface of the tetrapyd
    lb1, ub1 = interval_bounds[i1], interval_bounds[i1+1]   # upper and lower bounds of k1
    lb2, ub2 = interval_bounds[i2], interval_bounds[i2+1]   # upper and lower bounds of k1
    lb3, ub3 = interval_bounds[i3], interval_bounds[i3+1]   # upper and lower bounds of k1
    need_MC = (((lb1 + lb2 < ub3) & (ub1 + ub2 > lb3))      # The plane k1+k2-k3=0 intersects
                | ((lb2 + lb3 < ub1) & (ub2 + ub3 > lb1))   # The plane -k1+k2+k3=0 intersects
                | ((lb3 + lb1 < ub2) & (ub3 + ub1 > lb2)))  # The plane k1-k2+k3=0 intersects
    MC_count = np.sum(need_MC)

    # Draw samples uniformly within each cubic cell
    rng = default_rng(seed=0)
    r1 = rng.uniform(lb1[need_MC], ub1[need_MC], size=(MC_N_SAMPLES, MC_count))
    r2 = rng.uniform(lb2[need_MC], ub2[need_MC], size=(MC_N_SAMPLES, MC_count))
    r3 = rng.uniform(lb3[need_MC], ub3[need_MC], size=(MC_N_SAMPLES, MC_count))

    # Count the ratio of samples that lies inside the tetrapyd
    MC_weights = np.sum(((r1 + r2 >= r3) & (r2 + r3 >= r1) & (r3 + r1 >= r2)), axis=0) / MC_N_SAMPLES
    
    tetrapyd_weights[need_MC] *= MC_weights
    
    return grid, tetrapyd_weights



def poly_triplets_individual_degree(N):
    ''' List of triplets (p,q,r) such that
    N >= p >= q >= r >= 0 
    '''
    tuples = [[p, q, r] for p in range(N+1)
                for q in range(p+1)
                    for r in range(q+1)]
    
    return np.array(tuples).T

def poly_triplets_individual_degree_next_order(N):
    ''' List of triplets (p,q,r) such that
    N+1 = p >= q >= r >= 0 
    '''
    tuples = [[N+1, q, r] for q in range(N+2)
                    for r in range(q+1)]
    
    return np.array(tuples).T

def poly_triplets_total_degree(N):
    ''' List of triplets (p,q,r) such that
    p >= q >= r >= 0 and  p + q + r <= N
    '''

    # Increasing p, q within each n = p + q + r 
    #tuples = [[p, q, n-p-q] for n in range(N+1)
    #                 for p in range((n+2)//3, n+1)
    #                    for q in range((n-p+1)//2, min(p+1, n-p+1))]

    # Decreasing p, q within each n = p + q + r
    tuples = [[p, q, n-p-q] for n in range(N+1)
                     for p in range(n, (n+2)//3-1, -1)
                        for q in range(min(p, n-p), (n-p+1)//2-1, -1)]
    
    return np.array(tuples).T


def poly_triplets_total_degree_next_order(N):
    ''' List of triplets (p,q,r) such that
    p >= q >= r >= 0 and  p + q + r = N+1
    '''
    n = N + 1
    # Increasing p, q 
    # tuples = [[p, q, n-p-q] for p in range((n+2)//3, n+1)
    #                    for q in range((n-p+1)//2, min(p+1, n-p+1))]

    # Deacreasing p, q 
    tuples = [[p, q, n-p-q] for p in range(n, (n+2)//3-1, -1)
                        for q in range(min(p, n-p), (n-p+1)//2-1, -1)]
    
    return np.array(tuples).T


def gl_quadrature(alpha, N):
    # Gauss-Legendre quadrature on the interval [alpha, 1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(N)
    k_grid = alpha + (1 - alpha) / 2 * (gl_nodes + 1)
    k_weights = (1 - alpha) / 2 * gl_weights

    return k_grid, k_weights


def uni_tetra_triplets(alpha, N):

    k_grid = np.linspace(alpha, 1, N)

    tuples = [[i1, i2, i3] for i1 in range(N)
                for i2 in range(i1+1)
                    for i3 in range(i2+1)
                        if k_grid[i2] + k_grid[i3] >= k_grid[i1]]
    
    return np.array(tuples).T


def gl_tetra_triplets(alpha, N):

    k_grid, k_weights = gl_quadrature(alpha, N)

    tuples = [[i1, i2, i3] for i1 in range(N)
                for i2 in range(i1+1)
                    for i3 in range(i2+1)
                        if k_grid[i2] + k_grid[i3] >= k_grid[i1]]
    
    return np.array(tuples).T


def analytic_poly_integrals(ps, qs, rs, alpha):
    ''' Analytic values for the integrals of
    (x^p * y^q + z^r  + (5 syms)) / 6.0
    (or equivalently, x^p * y^q * z^r)
    over the tetrapyd.
    '''

    evals = 1 / ((1 + ps) * (1 + qs) * (1 + rs))
    evals -= gamma(1 + qs) * gamma(1 + rs) / ((3 + ps + qs + rs) * gamma(3 + qs + rs))
    evals -= gamma(1 + rs) * gamma(1 + ps) / ((3 + ps + qs + rs) * gamma(3 + rs + ps))
    evals -= gamma(1 + ps) * gamma(1 + qs) / ((3 + ps + qs + rs) * gamma(3 + ps + qs))
    evals *= 1 - alpha ** (3 + ps + qs + rs)

    return evals


def grid_poly_evaluations(grid_1d, ps, qs, rs, i1, i2, i3):
    ''' Computes the matrix containing the evaluations of
    f(x,y,z) = (x^p * y^q + z^r  + (5 syms)) / 6.0
    at the grid points.
    '''

    eval_p = np.power(grid_1d[np.newaxis,:], ps[:,np.newaxis])
    eval_q = np.power(grid_1d[np.newaxis,:], qs[:,np.newaxis])
    eval_r = np.power(grid_1d[np.newaxis,:], rs[:,np.newaxis])

    grid_evals = (eval_p[:,i1] * eval_q[:,i2] * eval_r[:,i3]
                    + eval_p[:,i1] * eval_q[:,i3] * eval_r[:,i2]
                    + eval_p[:,i2] * eval_q[:,i1] * eval_r[:,i3]
                    + eval_p[:,i2] * eval_q[:,i3] * eval_r[:,i1]
                    + eval_p[:,i3] * eval_q[:,i1] * eval_r[:,i2]
                    + eval_p[:,i3] * eval_q[:,i2] * eval_r[:,i1]) / 6.0

    return grid_evals

def poly_evaluations(p, q, r, k1, k2, k3):
    ''' Computes the matrix containing the evaluations of
    f(k1,k2,k3) = (k1^p * k2^q + k3^r  + (5 syms)) / 6.0
    '''

    evals = (k1 ** p * k2 ** q * k3 ** r
           + k1 ** p * k2 ** r * k3 ** q
           + k1 ** q * k2 ** p * k3 ** r
           + k1 ** q * k2 ** r * k3 ** p
           + k1 ** r * k2 ** p * k3 ** q
           + k1 ** r * k2 ** q * k3 ** p) / 6

    return evals


def orthonormal_polynomials(ps, qs, rs, alpha):
    ''' Orthonormalise the polynomials x^p y^q r^z on the unit tetrapyd (with alpha < 1).
        Uses modified Gram-Schmidt orthogonalisation based on the analytic integral values
        Returns a lower-triangluer matrix L such that the nth row specifies
        the orthgonal coefficients for the nth polynomial with (p,q,r) = (ps[n], qs[n], rs[n]).
    '''

    num_polys = ps.shape[0]

    # Cross inner product between polynomials
    I = np.zeros((num_polys, num_polys))

    for n in range(num_polys):
        p1, q1, r1 = ps[n], qs[n], rs[n]
        p2, q2, r2 = ps[:n+1], qs[:n+1], rs[:n+1]
        I[n,:n+1] = (analytic_poly_integrals(p1+p2, q1+q2, r1+r2, alpha)
                        + analytic_poly_integrals(p1+p2, q1+r2, r1+q2, alpha)
                        + analytic_poly_integrals(p1+q2, q1+p2, r1+r2, alpha)
                        + analytic_poly_integrals(p1+q2, q1+r2, r1+p2, alpha)
                        + analytic_poly_integrals(p1+r2, q1+p2, r1+q2, alpha)
                        + analytic_poly_integrals(p1+r2, q1+q2, r1+p2, alpha)
                            ) / 6
    
    I += I.T - np.diag(np.diag(I))   # Symmetrise

    C = np.zeros((num_polys, num_polys))
    N = np.zeros(num_polys)
    C[0,0] = 1
    N[0] = I[0,0] ** (-0.5)

    for n in range(num_polys):
        v = N[:n] ** 2 * np.matmul(C[:n,:n], I[n,:n])
        C[n,:n] = -np.matmul(v[:], C[:n,:n])
        C[n,n] = 1

        #N[n] = np.dot(I[n,:n+1], C[n,:n+1]) ** (-0.5)      # Faster but less accurate
        N[n] = np.dot(C[n,:n+1], np.matmul(I[:n+1,:n+1], C[n,:n+1])) ** (-0.5)      # Slower but more accurate

    L = C[:,:] * N[:,np.newaxis]

    return L
    
    
