import tetraquad
import numpy as np
import traceback

alpha = 0.001

#N_list = [30]
#N_list = np.arange(1, 31)
#N_list = [10]
#N_list = [i for i in range(1,50)] + [80]
N_list = [30] + [i for i in range(5,30)]
print(N_list)

#n_s = 0.9649
n_s = 0.96

for N in N_list:

    try:
        print("N = {}...".format(N))
#       filename = "outputs/tetraquad_alpha_N_{}.csv".format(N)
#       tetraquad.save_quadrature(filename, alpha, 1, N)
        filename = "outputs/tetraquad_alpha_negative_power_N_{}.csv".format(N)
        tetraquad.save_quadrature(filename, alpha, 1, N, negative_power=n_s-2)
#       filename = "outputs/uniform_quad_alpha_N_{}.csv".format(N)
#       tetraquad.save_uniform_quadrature(filename, alpha, 1, N, MC_N_SAMPLES=1000000)
    except Exception:
        traceback.print_exc()
