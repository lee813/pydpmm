# Copyright (c) 2017, Danyang 'Frank' Li <danyangl@mtu.edu>

from __future__ import division
from distribution import UnivariateGaussian
import numpy as np



class dpmm_gibbs_base(object):
    def __init__(self, init_K=5, x=[], alpha_prior=None):
        #Convert python array to numpy array
        self.x = np.asarray(x)
        self.K = init_K

        self.nn = np.ones(self.K)
        self.alpha_prior = alpha_prior
        self.alpha_0 = 1
            #np.random.gamma(self.alpha_prior['a'],self.alpha_prior['b'])

        # init zz randomly
        self.zz = np.random.randint(init_K, size=len(self.x))
        self.mu_0 = 1
        self.mu = np.ones(self.K)
        self.components = [mixture_component(ss=[], distn=UnivariateGaussian(mu=mu_i)) for mu_i in self.mu]

        for idx, c in enumerate(self.components):
            c.ss = self.x[self.zz == idx]
            self.nn[idx] = len(c.ss)

        self.n = len(self.x)


#TODO Add variance parameter
# fix var = 1
class direct_dpmm_gibbs(dpmm_gibbs_base):
    def __init__(self, init_K=5, x=[], alpha_prior=None):
        super(direct_dpmm_gibbs, self).__init__(init_K, x, alpha_prior)


    def new_component_probability(self, x):
        # TODO check formula
        return (1 / (2 * np.sqrt(np.pi))) * np.exp(- x**2 / 4)

    def new_component_log_integral(self, x):
        # TODO check formula
        return np.log(2 * np.sqrt(np.pi)) - (x**2/4)

    def sample_z(self):
        # STEP 2(d)
        # add z_i = new to form a new multi dist

        # Start sample aux indication variable z
        for idx, x_i in enumerate(self.x):
            kk = self.zz[idx]
            # Clean mixture components
            temp_zz, = np.where(self.zz == kk)
            temp_zz = np.setdiff1d(temp_zz, np.array([idx]))
            self.nn[kk] -= 1
            temp_ss = self.x[temp_zz]
            self.components[kk].ss = temp_ss
            if (len(temp_ss) == 0):
                #print('component deleted')
                self.components = np.delete(self.components, kk)
                self.K = len(self.components)
                self.nn = np.delete(self.nn, kk)
                zz_to_minus_1 = np.where(self.zz > kk)
                self.zz[zz_to_minus_1] -= 1

            proportion = np.array([])
            for k in range(0, self.K):
                # Calculate proportion for exist mixture component
                # Clean mixture components

                n_k = self.nn[k]
                #return exp
                _proportion = (n_k / (self.n + self.alpha_0 - 1)) * np.exp(self.components[k].distn.log_likelihood(x_i))
                proportion = np.append(proportion, _proportion)

            new_proportion = (self.alpha_0 / (self.n + self.alpha_0 - 1)) * self.new_component_probability(x_i)

            all_propotion = np.append(proportion, new_proportion)

            normailizedAllPropotion = all_propotion / sum(all_propotion)

            sample_z = np.random.multinomial(1, normailizedAllPropotion, size=1)

            z_index = np.where(sample_z == 1)[1][0]

            self.zz[idx] = z_index

            # found new component
            if (z_index == self.K):
                self.K += 1
                # sample new mu for new component
                # G_0 = n(0,1)
                new_mu = np.random.normal(0.5 * x_i, 0.5, 1);

                new_component = mixture_component(ss=[x_i], distn=UnivariateGaussian(mu=new_mu))

                self.components = np.append(self.components, new_component)
                self.nn = np.append(self.nn, 1)

                #print 'new component added'

            # add data to exist component
            else:
                self.components[z_index].ss = np.append(self.components[z_index].ss, x_i)
                self.nn[z_index] += 1

        for component in self.components:
            component.print_self()
        print('alpha -> ' + str(self.alpha_0))

    def sample_mu(self):

        for k in range(0, self.K):
            x_k = self.components[k].ss
            mu_k = np.random.normal((self.mu_0 + sum(x_k))/(1+len(x_k)), 1/(1 + len(x_k)), 1)
            self.components[k].distn.set_mu(mu=mu_k)
            #print('new mu -> ' + str(mu_k[0]))

    def sample_alpha_0(self):
        #Escobar and West 1995
        eta = np.random.beta(self.alpha_0 + 1,self.n,1)
        #Teh HDP 2005
        #construct the mixture model
        pi = self.n/self.alpha_0
        pi = pi/(1+pi)
        s = np.random.binomial(1,pi,1)
        #sample from a two gamma mixture models
        self.alpha_0 = np.random.gamma(self.alpha_prior['a'] + self.K - s, 1/(self.alpha_prior['b'] - np.log(eta)), 1)


class collapsed_dpmm_gibbs(dpmm_gibbs_base):
    def __init__(self, init_K=5, x=[], alpha_prior = None, observation_prior=None,):
        super(collapsed_dpmm_gibbs, self).__init__(init_K, x, alpha_prior)
        self.observation_prior = observation_prior

        #add a new empty component
        new_mu = np.random.normal(self.observation_prior['mu'], self.observation_prior['sigma'], 1);
        new_component = mixture_component(ss=[], distn=UnivariateGaussian(mu=new_mu))
        self.components = np.append(self.components, new_component)

        #print (UnivariateGaussian.epsilon_log_univariate_normal(self,-12,2) - UnivariateGaussian.epsilon_log_univariate_normal(self,1,1))


    def sample_z(self):
        for idx, x_i in enumerate(self.x):

            kk = self.zz[idx]
            # Clean mixture components
            temp_zz, = np.where(self.zz == kk)
            # print('----')
            # print len(self.components[kk].ss)
            temp_zz = np.setdiff1d(temp_zz,np.array([idx]))
            self.nn[kk] -= 1
            temp_ss = self.x[temp_zz]
            #print len(temp_ss)
            self.components[kk].ss = temp_ss
            if (len(temp_ss) == 0):
                #print('component deleted')
                #print(len(self.components))
                self.components = np.delete(self.components, kk)
                #print(len(self.components))
                self.K = len(self.components)
                self.nn = np.delete(self.nn,kk)
                zz_to_minus_1 = np.where(self.zz > kk)
                self.zz[zz_to_minus_1] -= 1

            pp = np.log(np.append(self.nn, self.alpha_0))

            for k in range(0, self.K):
                pp[k] = pp[k] + self.log_predictive(self.components[k],x_i)
                print(self.log_predictive(self.components[k],x_i))
            pp = np.exp(pp - np.max(pp))
            pp = pp/np.sum(pp)
            sample_z = np.random.multinomial(1, pp, size=1)
            print x_i


            z_index = np.where(sample_z == 1)[1][0]
            self.zz[idx] = z_index

            if(z_index == len(self.components) - 1):
                print('component added')
                new_mu = np.random.normal(0.5 * x_i, 0.5, 1);

                new_component = mixture_component(ss=[x_i], distn=UnivariateGaussian(mu=new_mu))

                self.components = np.append(self.components, new_component)
                self.K = len(self.components)
                self.nn = np.append(self.nn, 1)
            else:
                self.components[z_index].ss = np.append(self.components[z_index].ss, x_i)
                self.nn[z_index] += 1

        print '----Summary----'
        # print self.zz
        # print self.nn
        # for component in self.components:
        #     component.print_self()

    def sample_alpha_0(self):
        #Escobar and West 1995
        eta = np.random.beta(self.alpha_0 + 1,self.n,1)
        #Teh HDP 2005
        #construct the mixture model
        pi = self.n/self.alpha_0
        pi = pi/(1+pi)
        s = np.random.binomial(1,pi,1)
        #sample from a two gamma mixture models
        self.alpha_0 = np.random.gamma(self.alpha_prior['a'] + self.K - s, 1/(self.alpha_prior['b'] - np.log(eta)), 1)
        print self.alpha_0

    def log_predictive(self,component, x_i):
        ll = UnivariateGaussian.epsilon_log_univariate_normal(self, self.observation_prior['mu'] + np.sum(component.get_ss()) + x_i ,\
                                            self.observation_prior['sigma'] + component.get_n_k_minus_i() + 1) - \
             UnivariateGaussian.epsilon_log_univariate_normal(self, self.observation_prior['mu'] + np.sum(component.get_ss()), \
                                            self.observation_prior['sigma'] + component.get_n_k_minus_i())
        return ll



class mixture_component(object):
    def __init__(self, ss, distn):
        self.ss = ss
        self.distn = distn

        if(len(ss)> 0):
            self.n_k_minus_i = len(ss) - 1
        else:
            self.n_k_minus_i = 0

    def get_n_k_minus_i(self):
        if (len(self.ss) > 1):
            self.n_k_minus_i = len(self.ss) - 1
        else:
            self.n_k_minus_i = 0
        return self.n_k_minus_i
    def get_ss(self):
        return self.ss

    def print_self(self):
        print(self.ss)
        print('Mu: '+ str(self.distn.mu))