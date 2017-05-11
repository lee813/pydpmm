# Copyright (c) 2017, Danyang 'Frank' Li <danyangl@mtu.edu>

from __future__ import division
from distribution import UnivariateGaussian
import numpy as np
from scipy.stats import norm
from numpy.random import random


# fix var = 1
class dpmm_gibbs(object):
    def __init__(self, alpha_0=None, init_K=5, x=[]):
        self.alpha_0 = alpha_0
        self.x = x
        self.K = init_K

        self.mu_0 = 1
        self.mu = np.ones(self.K)
        self.z = np.ones((len(self.x), 1))

        #init ss
        ss_mtx = np.reshape(self.x, (5,4))

        self.components = [mixture_component(ss=[], distn=UnivariateGaussian(mu=mu_i)) for mu_i in self.mu]
        for idx,c in enumerate(self.components):
            c.ss = ss_mtx[idx,:]

        self.n = len(self.x)

    def sample_new_mu(self, x):
        return np.random.normal(0.5 * x, 0.5, 1)

    def new_component_probability(self, x):
        # TODO check formula
        return (1 / (2 * np.sqrt(np.pi))) * np.exp(-x ** 2 / 4)

    def sample_z(self):
        # STEP 2(d)
        # add z_i = new to form a new multi dist

        for component in self.components:
            print component.ss

         # Start sample aux indication variable z
        for idx, x_i in enumerate(self.x):



            proportion = np.array([])
            for k in range(0, self.K):
                # Calculate proportion for exist mixture component
                # Clean mixture components
                temp_ss = self.components[k].ss
                ss_delete_idx = []
                for idx,ss_i in enumerate(temp_ss):
                    if(ss_i == x_i):
                        ss_delete_idx.append(idx)
                temp_ss = np.delete(temp_ss, ss_delete_idx)
                self.components[k].ss = temp_ss
                if (len(temp_ss) == 0):
                    print('component deleted')
                    self.components = np.delete(self.components, k)
                    self.K = len(self.components)
                    break

                n_k = self.components[k].get_n_k_minus_i()
                _proportion = (n_k / (self.n + self.alpha_0 - 1)) * self.components[k].distn.log_likelihood(x_i)
                proportion = np.append(proportion, _proportion)

            pSum = sum(proportion)

            normailizedPropotion = proportion / pSum

            new_proportion = (self.alpha_0 / (self.n + self.alpha_0 - 1)) * self.new_component_probability(x_i)

            all_propotion = np.append(normailizedPropotion, new_proportion)

            aSum = sum(all_propotion)

            normailizedAllPropotion = all_propotion / aSum

            sample_z = np.random.multinomial(1, normailizedAllPropotion, size=1)

            z_index = np.where(sample_z == 1)[1][0]

            # found new component
            if (z_index == self.K):
                self.K += 1

                new_mu = np.random.normal(0.5 * x_i, 0.5, 1);

                new_component = mixture_component(ss=[x_i], distn=UnivariateGaussian(mu=new_mu))

                self.components = np.append(self.components, new_component)

                print 'new component added'

            # add data to exist component
            else:
                #print(z_index)
                #print(len(self.components))
                self.components[z_index].ss = np.append(self.components[z_index].ss, x_i)
                #print self.components[z_index].ss



    def sample_mu(self):

        for k in range(0, self.K):
            x_k = self.components[k].ss
            mu_k = np.random.normal((self.mu_0 + sum(x_k))/(1+len(x_k)), 1/(1 + len(x_k)), 1)
            self.components[k].distn.set_mu(mu=mu_k)
            print('new mu -> ' + str(mu_k[0]))


    def sample_alpha_0(self):
        pass


class mixture_component(object):
    def __init__(self, ss, distn):
        self.ss = ss
        self.distn = distn

        self.n = len(ss)
        if(self.n > 0):
            self.n_k_minus_i = len(ss) - 1
        else:
            self.n_k_minus_i = 0

    def get_n_k_minus_i(self):
        return len(self.ss) - 1
    def get_ss(self):
        return self.ss