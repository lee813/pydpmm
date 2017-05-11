from distribution import UnivariateGaussian
from gibbs import dpmm_gibbs
from matplotlib import pyplot as plt
import numpy as np
plt.style.use('ggplot')

alpha_0 = 1
x = [4.0429277,-3.71686209,0.73144389,5.05700962,4.70910861,1.38603028,-0.87114683,0.90842492,2.26485196,0.3287409, 1.85740593, -0.08981766,  0.11817958,  0.60973202,  1.88309994,
        1.47112954,  0.77061995,  1.24543065,  1.92506892,  0.7578275]


# mu = np.empty(len(x));
# loglikelihood = np.empty(len(x));
#
# gaussian = UnivariateGaussian(mu=2)
# result = test.rvs(10)
#
#
# plt.plot(result)
# plt.show()

##SAMPLE THETA

# for idx,xi in enumerate(x):
#     mu[idx] = gaussian.sample_new_mu(xi)

# for idx, (x_i, mu_i) in enumerate(zip(x, mu)):
#     loglikelihood[idx] = gaussian.log_likelihood(x_i,mu_i)

# plt.plot(x)
# plt.show()

# print(loglikelihood)


#sample = gaussian.sample_discrete(loglikelihood)

##Direct Gibbs sampling for DPMM
init_K = 5

gibbs = dpmm_gibbs(alpha_0,init_K,x)

iter = 20
for i in range(1,iter):
    gibbs.sample_z()
    gibbs.sample_mu()
