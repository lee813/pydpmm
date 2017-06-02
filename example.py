# Copyright (c) 2017, Danyang 'Frank' Li <danyangl@mtu.edu>
from gibbs import direct_dpmm_gibbs
from gibbs import collapsed_dpmm_gibbs
from matplotlib import pyplot as plt
plt.style.use('ggplot')

x = [4.0429277,10.71686209,10.73144389,5.05700962,4.70910861,1.38603028,-12.87114683,0.90842492,2.26485196,0.3287409, 1.85740593, -0.08981766,  0.11817958,  0.60973202,  1.88309994,
        1.47112954,  0.77061995,  1.24543065,  1.92506892,  0.7578275, -30.12442321]


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
#
# for idx, (x_i, mu_i) in enumerate(zip(x, mu)):
#     loglikelihood[idx] = gaussian.log_likelihood(x_i,mu_i)
#
# plt.plot(x)
# plt.show()

# print(loglikelihood)


#sample = gaussian.sample_discrete(loglikelihood)

##Direct Gibbs sampling for DPMM
init_K = 5
alpha_prior = {'a':1,'b':2}
observation_prior = {'mu':0,'sigma':10}

# gibbs = direct_dpmm_gibbs(init_K,x,alpha_prior)
#
# iter = 50
# for i in range(0,iter):
#     print('Iter: '+ str(i))
#     gibbs.sample_z()
#     gibbs.sample_mu()
#     gibbs.sample_alpha_0()

collapsed_gibbs = collapsed_dpmm_gibbs(init_K,x,alpha_prior,observation_prior)
iter = 50
for i in range(0,iter):
    collapsed_gibbs.sample_z()
    collapsed_gibbs.sample_alpha_0()
