import tetraquad
import numpy as np

alpha = 0.001

#N_list = [30]
N_list = np.arange(17,35)
#N_list = [10]
#N_list = [i for i in range(1,50)] + [80]
print(N_list)

for N in N_list:
    print("N = {}...".format(N))
#    filename = "outputs/tetraquad_alpha_N_{}.csv".format(N)
#    tetraquad.save_quadrature(filename, alpha, 1, N)
    filename = "outputs/uniform_quad_alpha_N_{}.csv".format(N)
    tetraquad.save_uniform_quadrature(filename, alpha, 1, N, MC_N_SAMPLES=1000000)
