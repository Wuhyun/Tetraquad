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
from scipy.special import gamma, beta, betainc, hyp2f1
from numpy.random import default_rng
import scipy
import pandas as pd
import mpmath
import matplotlib.pyplot as plt
from itertools import permutations

class Quadrature:
    ''' Class for storing the results of a 'Tetraquad' quadrature rule.
        Contains helper functions for handling quadrature objects.
    '''

    def __init__(self, k_min=None, k_max=None, quad_type="tetraquad", N=15, **kwargs):
        self.k_min = k_min
        self.k_max = k_max
        self.quad_type = quad_type
        self.N = N
        self.symmetrized = False

        if quad_type == "tetraquad":
            # Base "Tetraquad" quadrature
            self.alpha = k_min / k_max
            self.include_endpoints = kwargs.get("include_endpoints", True)
            self.N = N
            self.M = kwargs.get("M", None)

            # Compute the quadrature for k_max = 1 and rescale
            grid, weights = unit_quadrature_nnls(self.alpha, N, M=self.M, include_endpoints=self.include_endpoints)
            self.grid = k_max * grid
            self.weights = (k_max ** 3) * weights
        
        elif quad_type == "uniform":
            # Uniform quadrature - 3D trapezoidal rule
            self.alpha = k_min / k_max
            self.include_endpoints = kwargs.get("include_endpoints", False)
            self.include_borderline = kwargs.get("include_borderline", True)
            self.MC_n_samples = kwargs.get("MC_n_samples", 10000)
            self.use_COM = kwargs.get("use_COM", False)

            # Compute the quadrature for k_max = 1 and rescale
            grid, weights = uniform_tetrapyd_weights(self.alpha, N, MC_n_samples=self.MC_n_samples,
                                                     include_endpoints=self.include_endpoints,
                                                     include_borderline=self.include_borderline,
                                                     use_COM=self.use_COM)
            self.grid = k_max * grid
            self.weights = (k_max ** 3) * weights
        
        elif quad_type == "saved":
            # A dummy instance without the full information,
            # such as when it is loaded from a csv file
            # (Currently, the output file does not contain the full infromation.
            # TODO: use a better input/output file format and remove this)
            self.grid = kwargs.get("grid")
            self.weights = kwargs.get("weights")
            
    
    def integrate(self, integrand, symmetric_integrand=False):
        ''' Integrate the given function over a tetrapyd volume.
        '''

        if callable(integrand):
            # Function evaluation
            k1, k2, k3 = self.grid
            if symmetric_integrand or self.symmetrized:
                # Either the integrand or the quadrature is symmetrized
                evals = integrand(k1, k2, k3)
            else:
                # Otherwise, the integrand is symmetrized
                evals = (integrand(k1, k2, k3) + integrand(k1, k3, k2) + integrand(k2, k1, k3)
                         + integrand(k2, k3, k1) + integrand(k3, k1, k2) + integrand(k3, k2, k1)) / 6
        
        else:
            # "integrand" is assumed to be an array (or matrix)
            # of function(s)evaluated at the grid points
            # TODO: generalise to non-symmetric integrands
            assert symmetric_integrand or self.symmetrized
            evals = integrand

        result = np.dot(evals, self.weights)

        return result


    def integrate_multi(self, integrand_list):
        ''' Integrate the given functions over a tetrapyd volume.
        '''
        return np.array([self.integrate(integrand) for integrand in integrand_list])
    

    def symmetrize_quadrature(self):
        ''' Extends the quadrature based on a tetrapyd slice to the full tetrapyd by symmetry.
        '''

        df = self.to_dataframe()

        # create an empty list to store the rows of the new dataframe
        new_rows = []

        for index, row in df.iterrows():
            # generate all permutations of the row's values
            k1, k2, k3 = row["k1"], row["k2"], row["k3"]
            weight = row["weight"]

            if k1 == k2 and k2 == k3:
                # No need to symmetrize
                new_rows.append({'k1': k1, 'k2': k2, 'k3': k3, 'weight': weight})

            elif k1 == k2 or k2 == k3:
                # Need three copies
                new_rows.append({'k1': k1, 'k2': k2, 'k3': k3, 'weight': weight/3})
                new_rows.append({'k1': k2, 'k2': k3, 'k3': k1, 'weight': weight/3})
                new_rows.append({'k1': k3, 'k2': k1, 'k3': k2, 'weight': weight/3})
            
            else:
                # Need six copies
                perm = permutations([k1, k2, k3])
                for p in perm:
                    new_rows.append({'k1': p[0], 'k2': p[1], 'k3': p[2], 'weight': weight/6})

        # create the new dataframe from the list of rows
        new_df = pd.DataFrame(new_rows)
        new_grid = new_df[["k1", "k2", "k3"]].values.T
        new_weights = new_df[["weight"]].values.flatten()

        new_quad = Quadrature(self.k_min, self.k_max, "saved")
        new_quad.symmetrized = True
        new_quad.grid = new_grid
        new_quad.weights = new_weights

        return new_quad
    

    def to_dataframe(self):
        # Shows the grid and weights in a pandas dataframe
        grid, weights = self.grid, self.weights
        df = pd.DataFrame({"k1": grid[0], "k2": grid[1], "k3": grid[2], "weight": weights})
        #df = pd.DataFrame({"i1": i1, "i2": i2, "i3": i3, "k1": grid[0], "k2": grid[1], "k3": grid[2],
        #                    "weight": weights})
        return df

    def to_csv(self, output_path, float_format="%.18e"):
        # Saves the grid and weights in a csv format
        df = self.to_dataframe()
        df.to_csv(output_path, float_format=float_format)
    

    def from_csv(input_path):
        # Load a quadrature rule from a csv file 
        df = pd.read_csv(input_path)
        grid = np.stack([df["k1"], df["k2"], df["k3"]], axis=0)
        weights = df["weight"].to_numpy()
        quad = Quadrature(quad_type="saved", grid=grid, weights=weights)

        return quad


## Simple utility functions

def tetrapyd_volume(k_min, k_max):
    # Volume of the tetrapyd specified by k_min and k_max
    alpha = k_min / k_max
    return (0.5 - 3 * alpha ** 2 + 3 * alpha ** 3) * (k_max ** 3)


def poly_pairs_inidividual_degree(N):
    ''' List of pairs (p,q) such taht
    N >= p >= q >= 0
    '''
    pairs = [[p,q] for p in range(N+1)
                for q in range(p+1)]
    
    return np.array(pairs).T


def poly_pairs_total_degree(N):
    ''' List of pairs (p,q) such taht
    N >= p >= q >= 0 and p + q <= N
    '''
    pairs = [[p, n-p] for n in range(N+1)
                for p in range(n, (n-1)//2, -1)]
    
    return np.array(pairs).T


def poly_triplets_individual_degree(N, negative_power=None):
    ''' List of triplets (p,q,r) such that
    N >= p >= q >= r >= 0 
    If negative_power is not None,
    p, q, r are drawn from {negative_power, 0, 1, 2, ..., N-1}, 
    and p >= q >= r.
    '''

    if negative_power is None:
        tuples = [[p, q, r] for p in range(N+1)
                    for q in range(p+1)
                        for r in range(q+1)]
        
        return np.array(tuples).T

    else:
        res = poly_triplets_individual_degree(N, negative_power=None)
        res = res - 1.
        res[res < -0.5] = negative_power

        return res


def poly_triplets_individual_degree_next_order(N):
    ''' List of triplets (p,q,r) such that
    N+1 = p >= q >= r >= 0 
    '''
    tuples = [[N+1, q, r] for q in range(N+2)
                    for r in range(q+1)]
    
    return np.array(tuples).T


def poly_triplets_total_degree(N, negative_power=None):
    ''' List of triplets (p,q,r) such that
    p >= q >= r >= 0 and  p + q + r <= N
    If negative_power is not None, additionally include (p,q,r)s with
    p >= q >= 0, r = negative_power, and p + q <= N.
    '''

    if negative_power is None:
        # Increasing p, q within each n = p + q + r 
        #tuples = [[p, q, n-p-q] for n in range(N+1)
        #                 for p in range((n+2)//3, n+1)
        #                    for q in range((n-p+1)//2, min(p+1, n-p+1))]

        # Decreasing p, q within each n = p + q + r
        tuples = [[p, q, n-p-q] for n in range(N+1)
                        for p in range(n, (n+2)//3-1, -1)
                            for q in range(min(p, n-p), (n-p+1)//2-1, -1)]

    else: 
        tuples = [[0, 0, 0]]    # (0,0,0) always comes first
        tuples = tuples + [[p, n-p, negative_power] for n in range(N+1)
                                        for p in range(n, (n-1)//2, -1)]
        tuples = tuples + [[p, q, n-p-q] for n in range(1, N+1)
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


def poly_triplets_total_degree_ns(N, ns=0.9660499):
    ''' List of triplets (p,q,r) such that
    (p >= q >= r >= 0 and  p + q + r <= N) OR
    (p >= q >= 0 and r = ns-2 and p + q <= N).
    '''

    tuples = [[0, 0, 0]]

    # r = ns-2
    ps, qs = poly_pairs_total_degree(N)
    tuples = tuples + [[p, q, ns-2] for p, q in zip(ps, qs) if p + q >= 2] 

    # Decreasing p, q within each n = p + q + r
    tuples = tuples + [[p, q, n-p-q] for n in range(1, N+1)
                        for p in range(n, (n+2)//3-1, -1)
                            for q in range(min(p, n-p), (n-p+1)//2-1, -1)]
    
    return np.array(tuples).T


def uni_tetra_triplets(alpha, N, include_endpoints=True, include_borderline=True):
    ''' Indices for a uniform grid inside the tetrapyd
    satisfying 1 >= k1 >= k2 >= k3 >= alpha and k2 + k3 >= k1.
    When keep_borderline is True, keep all the grid points that lie outside
    the tetrapyd but has their voxel overlap with it.
    '''

    if include_endpoints:
        k_grid = np.linspace(alpha, 1, N)
    else:
        dk = (1 - alpha) / N
        k_grid = np.linspace(alpha+dk/2, 1-dk/2, N)
    

    if not include_borderline:
        # The following can miss out some volume elements
        tuples = [[i1, i2, i3] for i1 in range(N)
                    for i2 in range(i1+1)
                        for i3 in range(i2+1)
                            if k_grid[i2] + k_grid[i3] >= k_grid[i1]]
        i1, i2, i3 = np.array(tuples).T
    
    else:
        # 1D bounds for k grid points
        interval_bounds = np.zeros(N + 1)
        interval_bounds[0] = alpha
        interval_bounds[1:-1] = (k_grid[:-1] + k_grid[1:]) / 2
        interval_bounds[-1] = 1

        tuples = [[i1, i2, i3] for i1 in range(N)
                    for i2 in range(i1+1)
                        for i3 in range(i2+1)]
        i1, i2, i3 = np.array(tuples).T

        # Corners specifying the grid volume (voxel)
        lb1, ub1 = interval_bounds[i1], interval_bounds[i1+1]   # upper and lower bounds of k1
        lb2, ub2 = interval_bounds[i2], interval_bounds[i2+1]   # upper and lower bounds of k2
        lb3, ub3 = interval_bounds[i3], interval_bounds[i3+1]   # upper and lower bounds of k3
        inside = (k_grid[i2] + k_grid[i3] >= k_grid[i1])
        borderline = (((lb1 + lb2 < ub3) & (ub1 + ub2 > lb3))      # The plane k1+k2-k3=0 intersects
                    | ((lb2 + lb3 < ub1) & (ub2 + ub3 > lb1))   # The plane -k1+k2+k3=0 intersects
                    | ((lb3 + lb1 < ub2) & (ub3 + ub1 > lb2)))  # The plane k1-k2+k3=0 intersects
        keep = (inside | borderline)

        i1, i2, i3 = i1[keep], i2[keep], i3[keep]
    
    return i1, i2, i3, k_grid


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


def symmetric_polynomial(p, q, r):
    ''' Returns a function which represents
        f(k1,k2,k3) = (k1^p * k2^q + k3^r  + (5 syms)) / 6.0
    '''

    return lambda k1, k2, k3: poly_evaluations(p, q, r, k1, k2, k3)


def grid_sine_evaluations(grid_1d, omegas, phis, i1, i2, i3):
    ''' Computes the matrix containing the evaluations of
    f(x,y,z) = sin(omega * (x + y + z) + phi)
    at the grid points.
    '''

    eval_K = grid_1d[i1] + grid_1d[i2] + grid_1d[i3]
    grid_evals = np.sin(omegas[:,np.newaxis] * eval_K[np.newaxis,:] + phis[:,np.newaxis])

    return grid_evals


def sine_evaluations(omega, phi, k1, k2, k3):
    ''' Computes the matrix containing the evaluations of
    f(x,y,z) = sin(omega * (k1 * k2 * k3) + phi)
    '''

    evals = np.sin(omega * (k1 + k2 + k3) + phi)

    return evals


def sinusoidal(omega, phi):
    ''' Returns a function which represents
        f(k1,k2,k3) = sin(omega * (k1 + k2 + k3) + phi)
    '''
    return lambda k1, k2, k3: sine_evaluations(omega, phi, k1, k2, k3)


## Functions related to the analytic expression of
#  symmetric polynomials over the tetrapyd

def incomplete_beta(x, a, b):
    ''' The incomplete beta function B(x; a, b) as defined in https://dlmf.nist.gov/8.17
    Inputs a and b can be array-like, whereas x needs be a scalar.
    '''
    if np.abs(x) <= 1:
        # Used scipy's regularised incomplete beta function 'betainc'
        return beta(a, b) * betainc(a, b, x)
    else:
        # Use hyptergeometric function to compute the analytically-continued beta function
        # See Equation (8.17.7) of https://dlmf.nist.gov/8.17
        #return (x ** a) / a * hyp2f1(a, 1-b, a+1, x)
        #print("F({},{};{};{}) = {}".format(a+b, 1, a+1, x, hyp2f1(a+b,1,a+1,x)))
        
        #return ((0.+x) ** a) * ((1.-x) ** b) / a * hyp2f1(a+b, 1., a+1., x)
        return np.power(x, a) * np.power(1.-x, b) / a * hyp2f1(a+b, 1., a+1., x)


def generalised_incomplete_beta(x1, x2, a, b):
    ''' The generalised incomplete beta function 
    B(x1, x2; a, b) = B(x2; a, b) - B(x1; a, b)
    '''
    a, b = np.array(a), np.array(b)
    if x1 == 2:
        x1 = x1 + 1e-10     # Ad-hoc, avoid badness at x1=b=2

    int_ab = (a % 1 == 0) & (b % 1 == 0)
    if np.sum(int_ab) == a.size:
        # Powers a, b are integers
        return incomplete_beta(x2, a, b) - incomplete_beta(x1, a, b)
    else:
        result = np.zeros(a.size, dtype=complex) 

        # Normal method for integer powers
        if np.sum(int_ab) > 0:
            result[int_ab] = incomplete_beta(x2, a[int_ab], b[int_ab]) - incomplete_beta(x1, a[int_ab], b[int_ab])

        # Use mpmath for non-integer powers
        nint_ab = np.invert(int_ab)
        result[nint_ab] = np.array([complex(mpmath.betainc(av, bv, x1, x2)) for av, bv in zip(a[nint_ab], b[nint_ab])])

        return result


def analytic_sine_integrals(omegas, phis, alpha):
    ''' Analytic values for the integrals of
    f(x,y,z) = sin(omega * (x + y + z) + phi)
    over the tetrapyd.
    '''

    fact =  1 / (4 * omegas ** 3)
    evals = (9 * np.cos(phis + 2 * omegas) + 4 * np.cos(phis + 3 * omegas)
             -4 * np.cos(phis + 3 * alpha * omegas) + 3 * np.cos(phis + 4 * alpha * omegas)
             -12 * np.cos(phis + (2 + alpha) * omegas) + 6 * omegas * np.sin(phis + 2 * omegas)
             -12 * alpha * omegas * np.sin(phis + 2 * omegas))
    evals *= fact

    return evals


def analytic_poly_integrals(ps, qs, rs):
    ''' Analytic values for the integrals of
    (x^p * y^q + z^r  + (5 syms)) / 6.0
    (or equivalently, x^p * y^q * z^r)
    over the tetrapyd with alpha = 0 (bounded below by zero).
    '''

    evals = 1 / ((1 + ps) * (1 + qs) * (1 + rs))
    evals -= gamma(1 + qs) * gamma(1 + rs) / ((3 + ps + qs + rs) * gamma(3 + qs + rs))
    evals -= gamma(1 + rs) * gamma(1 + ps) / ((3 + ps + qs + rs) * gamma(3 + rs + ps))
    evals -= gamma(1 + ps) * gamma(1 + qs) / ((3 + ps + qs + rs) * gamma(3 + ps + qs))

    return evals


def analytic_poly_integrals_alpha(ps, qs, rs, alpha):
    ''' Analytic values for the integrals of
    (x^p * y^q + z^r  + (5 syms)) / 6.0
    (or equivalently, x^p * y^q * z^r)
    over the tetrapyd, bounded below by alpha, assumed to be less than 1/2.
    See Wuhyun's notes for the derivation of this formula.
    '''

    if alpha == 0:
        return analytic_poly_integrals(ps, qs, rs)
    
    if isinstance(ps, int) or isinstance(ps, float):
        ps, qs, rs = np.array([ps]), np.array([qs]), np.array([rs])
    else:
        ps, qs, rs = np.array(ps), np.array(qs), np.array(rs)

    a = alpha
    b = 1 - a

    evals = np.zeros(len(ps))

    # Prevent overflow errors from large p+q+r
    ignore_alpha = ((ps+qs+rs) > np.log(1e-200) / np.log(alpha))
    evals[ignore_alpha] = analytic_poly_integrals(ps[ignore_alpha], qs[ignore_alpha], rs[ignore_alpha])

    do_alpha = np.invert(ignore_alpha)
    p, q, r = ps[do_alpha], qs[do_alpha], rs[do_alpha]

    evals_alpha = np.zeros(len(do_alpha))

    # alpha^(p+1)
    ap1, aq1, ar1 = a ** (p+1), a ** (q+1), a ** (r+1)
    # beta^(p+1)
    bp1, bq1, br1 = b ** (p+1), b ** (q+1), b ** (r+1)

    # B(x1, x2; a, b)
    B = generalised_incomplete_beta

    # Integral over the cube [alpha,1]^3
    I_cube = (1-ap1) * (1-aq1) * (1-ar1) / ((p+1.) * (q+1.) * (r+1.))

    # Integral over the volume inside the cube but outside the tetrapyd,
    # where x >= y+z
    I_x = B(a, b, q+1, r+1) / (p+q+r+3.) / (q+r+2.)
    I_x -= (((r+1.) * ar1 * bq1 / (q+r+2.)) + ((q+1.) * aq1 * br1 / (q+r+2.)) - aq1 * ar1) / ((p+1.) * (q+1.) * (r+1.))
    #I_x += a**(p+q+r+3) * ((-1)**q * B(2, 1./a, p+2, q+1) + (-1)**r * B(2, 1./a, p+2, r+1)) / ((p+1.) * (p+q+r+3.))
    I_x += a**(p+q+r+3) * np.real((-1.+0j)**q * B(2, 1/a, p+2, q+1) + (-1.+0j)**r * B(2, 1/a, p+2, r+1)) / ((p+1.) * (p+q+r+3.))

    # Same for y >= z+x
    I_y = B(a, b, r+1, p+1) / (p+q+r+3.) / (r+p+2.)
    I_y -= (((p+1.) * ap1 * br1 / (r+p+2.)) + ((r+1.) * ar1 * bp1 / (r+p+2.)) - ar1 * ap1) / ((p+1.) * (q+1.) * (r+1.))
    #I_y += a**(p+q+r+3) * ((-1)**r * B(2, 1./a, q+2, r+1) + (-1)**p * B(2, 1./a, q+2, p+1)) / ((q+1.) * (p+q+r+3.))
    I_y += a**(p+q+r+3) * np.real((-1.+0j)**r * B(2, 1/a, q+2, r+1) + (-1.+0j)**p * B(2, 1/a, q+2, p+1)) / ((q+1.) * (p+q+r+3.))

    # Same for z >= x_y
    I_z = B(a, b, p+1, q+1) / (p+q+r+3.) / (p+q+2.)
    I_z -= (((q+1.) * aq1 * bp1 / (p+q+2.)) + ((p+1.) * ap1 * bq1 / (p+q+2.)) - ap1 * aq1) / ((p+1.) * (q+1.) * (r+1.))
    #I_z += a**(p+q+r+3) * ((-1)**p * B(2, 1./a, r+2, p+1) + (-1)**q * B(2, 1./a, r+2, q+1)) / ((r+1.) * (p+q+r+3.))
    I_z += a**(p+q+r+3) * np.real((-1.+0j)**p * B(2, 1/a, r+2, p+1) + (-1.+0j)**q * B(2, 1/a, r+2, q+1)) / ((r+1.) * (p+q+r+3.))

    evals_alpha = I_cube - I_x - I_y - I_z

    evals[do_alpha] = np.real(evals_alpha[:])

    return evals


def analytic_poly_cross_product_alpha(ps, qs, rs, alpha):
    ''' Returns a matrix containing cross inner products of
    (x^p * y^q + z^r  + (5 syms)) / 6.0
    over the tetrapyd domain bounded below by alpha (<0.5).
    Optimised for this special purpose.
    '''

    # Arrays need be int for indexing purposes
    ps = np.round(ps).astype(int)
    qs = np.round(qs).astype(int)
    rs = np.round(rs).astype(int)

    if alpha == 0:
        # When k_min = 0, the calculations are more straightforward

        num_polys = len(ps)
        cross_prod = np.zeros((num_polys, num_polys))

        psv, qsv, rsv = ps[:,np.newaxis], qs[:,np.newaxis], rs[:,np.newaxis]
        perms = [(psv+ps, qsv+qs, rsv+rs), (psv+ps, qsv+rs, rsv+qs),
                 (psv+qs, qsv+ps, rsv+rs), (psv+qs, qsv+rs, rsv+ps),
                 (psv+rs, qsv+ps, rsv+qs), (psv+rs, qsv+qs, rsv+ps)]
        
        for p, q, r in perms:
            cross_prod += analytic_poly_integrals(p, q, r) / 6

        return cross_prod

    # Shorthand definitions
    a = alpha
    b = 1 - a
    p_max = np.max(ps)
    max_power = 2 * p_max + 2

    # Precompute B(a, b; p, q) for 1 <= p, q <= max_power
    # Since a + b = 1, B(a, b; p, q) is symmteric in p and q
    pre_beta_a_b = np.zeros((max_power+1, max_power+1))
    for p in range(1, max_power+1):
        # Scipy's normalised incomplete beta function
        q = np.arange(1, p+1)
        pre_beta_a_b[p,1:p+1] = beta(p, q) * (betainc(p, q, b) - betainc(p, q, a))
    # Symmetrise matrix
    pre_beta_a_b += pre_beta_a_b.T - np.diag(np.diag(pre_beta_a_b))

    # Precompute (a**(p+q) * (-1)**q) * B(2, 1/a; p, q)
    cut = int(np.ceil(np.log(1e-200) / np.log(a)))   # Ignore this term for p+q > cut
    pre_beta_2_ainv = np.zeros((max_power+1, max_power+1))
    for p in range(1, min(max_power+1, cut)):
        f2, fainv = np.power(2.*a, p) / p, 1. / p
        #B(x;p,q) = np.power(x, p) * np.power(1.-x, q) / p * hyp2f1(p+q, 1., p+1., x)
        # Scipy's hypergeometric function
        q = np.arange(1, min(max_power+1, cut+1-p))
        pre_beta_2_ainv[p,q] = (fainv * np.power(1.-a, q) * hyp2f1(p+q, 1, p+1, 1/a)
                                    - f2 * np.power(a, q) * hyp2f1(p+q, 1, p+1, 2))

    #print("Betas precomputed")

    num_polys = len(ps)
    cross_prod = np.zeros((num_polys, num_polys))

    for n in range(num_polys):
        p1, q1, r1 = ps[n], qs[n], rs[n]
        p2, q2, r2 = ps[:n+1], qs[:n+1], rs[:n+1]
        perms = [(p1+p2, q1+q2, r1+r2), (p1+p2, q1+r2, r1+q2),
                 (p1+q2, q1+p2, r1+r2), (p1+q2, q1+r2, r1+p2),
                 (p1+r2, q1+p2, r1+q2), (p1+r2, q1+q2, r1+p2)]

        for p, q, r in perms:
            # alpha^(p+1)
            ap1, aq1, ar1 = a ** (p+1), a ** (q+1), a ** (r+1)
            # beta^(p+1)
            bp1, bq1, br1 = b ** (p+1), b ** (q+1), b ** (r+1)

            # Integral over the cube [alpha,1]^3
            I_cube = (1-ap1) * (1-aq1) * (1-ar1) / ((p+1.) * (q+1.) * (r+1.))

            # Integral over the volume inside the cube but outside the tetrapyd,
            # where x >= y+z
            I_x = pre_beta_a_b[q+1,r+1] / (p+q+r+3.) / (q+r+2.)
            I_x -= (((r+1.) * ar1 * bq1 / (q+r+2.)) + ((q+1.) * aq1 * br1 / (q+r+2.)) - aq1 * ar1) / ((p+1.) * (q+1.) * (r+1.))
            #I_x += a**(p+q+r+3) * ((-1) * pre_beta_2_ainv[p+2,q+1] + (-1) * pre_beta_2_ainv[p+2,r+1]) / ((p+1.) * (p+q+r+3.))
            I_x -= (ar1 * pre_beta_2_ainv[p+2,q+1] + aq1 * pre_beta_2_ainv[p+2,r+1]) / (a * (p+1.) * (p+q+r+3.))

            # Same for y >= z+x
            I_y = pre_beta_a_b[r+1,p+1] / (p+q+r+3.) / (r+p+2.)
            I_y -= (((p+1.) * ap1 * br1 / (r+p+2.)) + ((r+1.) * ar1 * bp1 / (r+p+2.)) - ar1 * ap1) / ((p+1.) * (q+1.) * (r+1.))
            #I_y += a**(p+q+r+3) * ((-1) * pre_beta_2_ainv[q+2,r+1] + (-1) * pre_beta_2_ainv[q+2,p+1]) / ((q+1.) * (p+q+r+3.))
            I_y -= (ap1 * pre_beta_2_ainv[q+2,r+1] + ar1 * pre_beta_2_ainv[q+2,p+1]) / (a * (q+1.) * (p+q+r+3.))

            # Same for z >= x_y
            I_z = pre_beta_a_b[p+1,q+1] / (p+q+r+3.) / (p+q+2.)
            I_z -= (((q+1.) * aq1 * bp1 / (p+q+2.)) + ((p+1.) * ap1 * bq1 / (p+q+2.)) - ap1 * aq1) / ((p+1.) * (q+1.) * (r+1.))
            #I_z += a**(p+q+r+3) * ((-1) * pre_beta_2_ainv[r+2,p+1] + (-1) * pre_beta_2_ainv[r+2,q+1]) / ((r+1.) * (p+q+r+3.))
            I_z -= (aq1 * pre_beta_2_ainv[r+2,p+1] + ap1 * pre_beta_2_ainv[r+2,q+1]) / (a * (r+1.) * (p+q+r+3.))

            cross_prod[n,:n+1] += (I_cube - I_x - I_y - I_z) / 6

    # Symmetrise
    cross_prod += cross_prod.T - np.diag(np.diag(cross_prod))

    return cross_prod


## Main quadrature code

def unit_quadrature_nnls(alpha, N, M=None, include_endpoints=True):
    ''' Returns a quadrature rule which has N grid points on each dimension.
    Minimises the integration error of symmetric polynomials 
    order <= f(N) over a tetrapyd specified by the triangle conditions and
    alpha <= k1, k2, k3 <= 1
    '''

    # Set up grid points
    i1, i2, i3, grid_1d = uni_tetra_triplets(alpha, N, include_endpoints, include_borderline=False)
    num_weights = i1.shape[0]

    grid = np.array([grid_1d[i1], grid_1d[i2], grid_1d[i3]])

    # Prepare orthogonal polynomials
    if M is None:
        #M = 2 * N
        M = round(1.5 * N)
    ps, qs, rs = poly_triplets_total_degree(M)
    num_polys = ps.shape[0]

    #print("M =", M, ", N =", N)

    # Obtain orthonormalisation coefficients for the polynomials
    ortho_L = orthonormal_polynomials(ps, qs, rs, alpha)

    poly_ampl = np.copy(np.diag(ortho_L))
    ortho_L /= np.sqrt(poly_ampl)[:,np.newaxis]

    # Evaluations of the polynomials at grid points
    grid_evals = grid_poly_evaluations(grid_1d, ps, qs, rs, i1, i2, i3)
    grid_evals = np.matmul(ortho_L, grid_evals)

    # We are now ready to compute the quadrature weights.
    # Non-Negative Least Squares:
    # minimise ||A x - b||^2 for non-negative x

    print(num_polys, num_weights)

    A = grid_evals
    tetra_volume = 0.5 - 3 * alpha ** 2 + 3 * alpha ** 3
    b = np.concatenate([[ortho_L[0,0]*tetra_volume], np.zeros(num_polys-1)])

    fact = np.sqrt(np.sum(A * A, axis=1))
    A = A / fact[:,np.newaxis]
    b = b / fact

    x, rnorm = scipy.optimize.nnls(A, b)
    print("NNLS complete, rnorm {}".format(rnorm))
    weights = x.flatten()

    nonzero_weights = np.nonzero(weights)[0]
    print("Out of {} weights, {} of them are nonzero".format(len(weights), len(nonzero_weights)))
    grid = grid[:, nonzero_weights]
    weights = weights[nonzero_weights]

    return grid, weights


def uniform_tetrapyd_weights(alpha, N, MC_n_samples=10000, include_endpoints=True, include_borderline=True, use_COM=False):
    ''' Returns a uniform tetrapyd grid with weights proportional to 
    the volume of the grid point (voxel).
    If use_COM is True, the grid points are located at the center of mass (COM) of each voxel.
    '''

    # Set up grid points
    i1, i2, i3, grid_1d = uni_tetra_triplets(alpha, N, include_endpoints, include_borderline=include_borderline)
    num_weights = i1.shape[0]

    grid = np.zeros((3, num_weights))
    grid[0,:] = grid_1d[i1]
    grid[1,:] = grid_1d[i2]
    grid[2,:] = grid_1d[i3]

    # 1D bounds and weights for k grid points
    interval_bounds = np.zeros(N + 1)
    interval_bounds[0] = alpha
    interval_bounds[1:-1] = (grid_1d[:-1] + grid_1d[1:]) / 2
    interval_bounds[-1] = 1
    k_weights = np.diff(interval_bounds)

    # Initialise weights based on symmetry and grid intervals
    tetrapyd_weights = k_weights[i1] * k_weights[i2] * k_weights[i3]
    tetrapyd_weights[(i1 != i2) & (i2 != i3)] *= 6   # Distinct indices
    tetrapyd_weights[(i1 != i2) & (i2 == i3)] *= 3   # Two identical indices
    tetrapyd_weights[(i1 == i2) & (i2 != i3)] *= 3   # Two identical indices

    # Further weights for points of the surface of the tetrapyd
    lb1, ub1 = interval_bounds[i1], interval_bounds[i1+1]   # upper and lower bounds of k1
    lb2, ub2 = interval_bounds[i2], interval_bounds[i2+1]   # upper and lower bounds of k2
    lb3, ub3 = interval_bounds[i3], interval_bounds[i3+1]   # upper and lower bounds of k3
    need_MC = (((lb1 + lb2 < ub3) & (ub1 + ub2 > lb3))      # The plane k1+k2-k3=0 intersects
                | ((lb2 + lb3 < ub1) & (ub2 + ub3 > lb1))   # The plane -k1+k2+k3=0 intersects
                | ((lb3 + lb1 < ub2) & (ub3 + ub1 > lb2)))  # The plane k1-k2+k3=0 intersects
    MC_count = np.sum(need_MC)

    # Draw samples uniformly within each cubic cell
    rng = default_rng(seed=0)
    r1 = rng.uniform(lb1[need_MC], ub1[need_MC], size=(MC_n_samples, MC_count))
    r2 = rng.uniform(lb2[need_MC], ub2[need_MC], size=(MC_n_samples, MC_count))
    r3 = rng.uniform(lb3[need_MC], ub3[need_MC], size=(MC_n_samples, MC_count))

    # Estimate the fraction of samples that lie inside the tetrapyd
    #in_tetra = ((r1 >= r2) & (r2 >= r3) & (r2 + r3 >= r1))
    in_tetra = ((r1 + r2 >= r3) & (r2 + r3 >= r1) & (r3 + r1 >= r2))
    MC_weights = np.sum(in_tetra, axis=0) / MC_n_samples
 
    tetrapyd_weights[need_MC] *= MC_weights

    if use_COM:
        n_in_tetra = np.sum(in_tetra, axis=0)
        grid[0,need_MC] = np.sum(r1 * in_tetra, axis=0) / n_in_tetra
        grid[1,need_MC] = np.sum(r2 * in_tetra, axis=0) / n_in_tetra
        grid[2,need_MC] = np.sum(r3 * in_tetra, axis=0) / n_in_tetra

    # Make sure the overall volume is exact
    #tetra_volume = 0.5 - 3 * alpha ** 2 + 3 * alpha ** 3
    #tetrapyd_weights *= tetra_volume / np.sum(tetrapyd_weights)
    
    return grid, tetrapyd_weights


def orthonormal_polynomials(ps, qs, rs, alpha):
    ''' Orthonormalise the polynomials x^p y^q r^z on the unit tetrapyd (with alpha < 1).
        Uses modified Gram-Schmidt orthogonalisation based on the analytic integral values.
        Returns a lower-triangluer matrix L such that the nth row specifies
        the orthgonal coefficients for the nth polynomial with (p,q,r) = (ps[n], qs[n], rs[n]).
    '''

    num_polys = ps.shape[0]

    # Cross inner product between polynomials
    I = analytic_poly_cross_product_alpha(ps, qs, rs, alpha)
    #print("I=", I)

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


## Testing routines

def test_polynomial_integration(quadratures, alpha=None, max_degree=10):
    # Compare the numerical integral errors of the symmetric polynomials
    # over the tetrapyd with the analytical expression

    if not isinstance(quadratures, list):
        quadratures = [quadratures]
    
    if alpha is None:
        alpha = quadratures[0].alpha

    ps, qs, rs = poly_triplets_total_degree(max_degree)
    polys = [symmetric_polynomial(p, q, r) for p, q, r in zip(ps, qs, rs)]

    analytic = analytic_poly_integrals_alpha(ps, qs, rs, alpha)
    numericals = [quad.integrate_multi(polys) for quad in quadratures]
    
    errors = [(numerical-analytic)/analytic for numerical in numericals]
    
    if len(errors) == 1:
        errors = errors[0]

    return errors


def plot_test_polynomial_integration(quadratures, alpha=None, labels=None, colors=None, max_degree=30, fig=None, ax=None):
    # Create a plot for comparing the numerical integral errors of
    # the symmetric polynomials over the tetrapyd with the analytical expression

    if not isinstance(quadratures, list):
        quadratures = [quadratures]
    n_quads = len(quadratures)

    if fig is None:
        fig, ax = plt.subplots()
    
    if labels is None:
        labels = [f"Quad #{n+1}" for n in range(n_quads)]

    # Integration Errors
    errors = test_polynomial_integration(quadratures, alpha, max_degree)

    # Polynomials used
    ps, qs, rs = poly_triplets_total_degree(max_degree)
    tot_degree = ps + qs + rs
    poly_number = np.arange(len(tot_degree))

    # Helper functions for the plot
    def forward(x):
        return np.interp(x, poly_number, tot_degree)
    def inverse(y):
        return np.interp(y, tot_degree, poly_number)

    # Plot results
    if colors is None:
        for error, label in zip(errors, labels):
            ax.plot(np.abs(error), label=label)
    else:
        for error, label, color in zip(errors, labels, colors):
            ax.plot(np.abs(error), label=label, c=color)

    # Label axes
    ax.set_xlabel("Polynomial Number")
    ax.set_xlim([0,len(poly_number)])
    ax.set_yscale('log')
    ax.set_ylabel("Fractional Error")
    secax = ax.secondary_xaxis("top", functions=(forward,inverse))
    secax.set_xlabel("Total Order $(p+q+r)$")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return fig, ax