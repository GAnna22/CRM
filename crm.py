# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:22:27 2021

@author: Anna.Gubanova
"""

import numpy as np
import scipy.optimize

class CRM_general:
    def __init__(self, t_0, t_n, dt, injection_rates, production_rates,
                 crm_type='CRMP_pwf', pwf=None, all_weights=None):
        """Main class of general CRM.

        Parameters
        ----------
        t_0: int, start of training (optimization) period
        t_n: int, end of training period
        dt: array_like, difference between timesteps of measurements
        injection_rates: array_like, shape (Nt, NI), injection history of
                         all injectors
        production_rates: array_like, shape (Nt, NP), production history
                          among all producing wells
        crm_type: str, {'CRMT', 'CRMP', 'CRMIP',
                        'CRMT_pwf', 'CRMP_pwf', 'CRMIP_pwf'}
            Type of solver. Should be one of

                - 'CRMT' - CRM Tank: 4 parameters for the Field
                - 'CRMP' - Producer-based CRM: NP*(NI + 2) parameters
                - 'CRMIP' - Injector-Producer Pair-based CRM: NP*(1 + 2*NI)
                            parameters
                - 'CRMT_pwf', 'CRMP_pwf', 'CRMIP_pwf' - same models with
                   accounting for bottom-hole pressure variation
        pwf: array-like, shape (Nt, NP), bottomhole pressure history of all
             producing wells (+ 1/NP/NI*NP parameters depending on crm_type)
        all_weights: array_like (n_weights,), pretrained coefficient weights
                     or custom initial guess
            The order of weights is:
                (tau_prime, tau_weights, lambda_weights, (J_pwf))
        """
        self.t_0 = t_0
        self.t_n = t_n
        self.dt = dt
        self.dt_cum = dt.cumsum()
        self.production_rates = production_rates
        self.prod_num = production_rates.shape[1]
        self.injection_rates = injection_rates[:, :, None].repeat(self.prod_num, axis=-1)
        self.inj_num = injection_rates.shape[1]
        self.mask = production_rates == 0.
        zero_rows, zero_cols = np.where(self.mask)
        self.dpwf = np.vstack((pwf[0], pwf[1:] - pwf[:-1]))
        self.dpwf[zero_rows, zero_cols] = 0.
        self.dpwf[zero_rows+1, zero_cols] = 0.
        self.injection_rates[zero_rows, :, zero_cols] = 0.
        self.crm_type = crm_type
        if crm_type[:4] in ['CRMT', 'CRMP', 'CRMI']:
            if 'pwf' in crm_type:
                self.liquid_rate_calc = self.CRMIPT_pwf
            else:
                self.liquid_rate_calc = self.CRMIPT
        self.error_func = []
        self.init_weights(all_weights)

    def set_injection(self, injection_rates):
        self.injection_rates = injection_rates

    def set_production(self, production_rates):
        self.production_rates = production_rates

    def init_weights(self, all_weights=None):
        """Initializing weights of CRModel with random numbers with uniform
           distribution or with given weights."""
        if all_weights is None:
            self.tau_prime = np.random.uniform(0., 10., self.prod_num)
            self.lambda_weights = np.random.uniform(0., 1., (self.inj_num, self.prod_num))
            self.lambda_weights /= self.lambda_weights.sum(1)[:, None]
            if self.crm_type[:4] in ['CRMP', 'CRMT']:
                self.tau_weights = np.random.uniform(0., 10., (1, self.prod_num))
                self.all_weights = np.hstack((self.tau_prime, self.tau_weights.flatten(),
                                              self.lambda_weights.flatten()))
                if 'pwf' in self.crm_type:
                    self.J_pwf = np.random.uniform(0., 10., (1, self.prod_num))
                    self.all_weights = np.hstack((self.all_weights, self.J_pwf.flatten()))
            elif self.crm_type[:5] == 'CRMIP':
                self.tau_weights = np.random.uniform(0., 10., (self.inj_num, self.prod_num))
                self.all_weights = np.hstack((self.tau_prime, self.tau_weights.flatten(),
                                              self.lambda_weights.flatten()))
                if 'pwf' in self.crm_type:
                    self.J_pwf = np.random.uniform(0., 10., (self.inj_num, self.prod_num))
                    self.all_weights = np.hstack((self.all_weights, self.J_pwf.flatten()))
        else:
            self.all_weights = all_weights
            self.tau_prime = all_weights[:self.prod_num]
            if self.crm_type[:4] in ['CRMP', 'CRMT']:
                self.tau_weights = all_weights[self.prod_num:2*self.prod_num].reshape(1, self.prod_num)
                self.lambda_weights = all_weights[2*self.prod_num:
                                                  2*self.prod_num+
                                                  self.inj_num*self.prod_num].reshape(self.inj_num, self.prod_num)
                if 'pwf' in self.crm_type:
                    self.J_pwf = all_weights[-self.prod_num:].reshape(1, self.prod_num)
            elif self.crm_type[:5] == 'CRMIP':
                self.tau_weights = all_weights[self.prod_num:
                                               self.prod_num*(1+self.inj_num)].reshape(self.inj_num, self.prod_num)
                self.lambda_weights = all_weights[self.prod_num*(1+self.inj_num):
                                                  self.prod_num*(1+2*self.inj_num)].reshape(self.inj_num, self.prod_num)
                if 'pwf' in self.crm_type:
                    self.J_pwf = all_weights[-self.inj_num*self.prod_num:].reshape(self.inj_num, self.prod_num)

    def CRMIPT(self, t_0, t_curr):
        """Function to optimize without accounting for pressure drops."""
        conv_inj = 0.
        for t_m in range(t_0+1, t_curr+1):
            exp_term = ((1 - np.exp(-self.dt[t_m]/self.tau_weights))*
                         np.exp((self.dt_cum[t_m] - self.dt_cum[t_curr])/self.tau_weights))
            conv_inj += (exp_term*self.injection_rates[t_m]*(1. + np.sum(self.mask[t_m]*self.lambda_weights, axis=1,
                                                                         keepdims=True))
                        )
        return (self.production_rates[t_0]*np.exp(-(self.dt_cum[t_curr]-self.dt_cum[t_0])/self.tau_prime) +
                (self.lambda_weights*conv_inj,).sum(0))

    def CRMIPT_pwf(self, t_0, t_curr):
        """Function to optimize with accounting for pressure drops."""
        conv_inj = 0.
        conv_pwf = 0.
        for t_m in range(t_0+1, t_curr+1):
            exp_term = ((1 - np.exp(-self.dt[t_m]/self.tau_weights))*
                         np.exp((self.dt_cum[t_m] - self.dt_cum[t_curr])/self.tau_weights))
            conv_inj += (exp_term*self.injection_rates[t_m]*(1. + np.sum(self.mask[t_m]*self.lambda_weights, axis=1,
                                                                         keepdims=True))
                        )
            conv_pwf += (exp_term*self.tau_weights*self.J_pwf).sum(0)*self.dpwf[t_m]/self.dt[t_m]
        return (self.production_rates[t_0]*np.exp(-(self.dt_cum[t_curr]-self.dt_cum[t_0])/self.tau_prime) +
                (self.lambda_weights*conv_inj).sum(0) - conv_pwf)

    def residuals(self, all_weights):
        """Residuals between predicitons and historical data."""
        self.init_weights(all_weights)
        preds = np.array([self.liquid_rate_calc(self.t_0, t_curr) for t_curr in range(self.t_0, self.t_n+1)])
        return (preds - self.production_rates[self.t_0:self.t_n+1]).flatten()

    def objective_func(self, all_weights):
        """MAE objective function to be minimized."""
        self.init_weights(all_weights)
        preds = np.array([self.liquid_rate_calc(self.t_0, t_curr) for t_curr in range(self.t_0, self.t_n+1)])
        self.error_func.append(
            ((preds - self.production_rates[self.t_0:self.t_n+1])**2).sum()/np.prod(preds.shape)
        ) #+ (self.mu_weights*(self.lambda_weights.sum(1) - 1)).sum()
        return self.error_func[-1]

    def optimize(self, dtype='LSq'):
        """Optimization function. Constrained and unconstrained minimization.

        Parameters
        ----------
        dtype : str, {'N-M', 'SLSQP', 'BFGS', 'DE', 'anneal', 'LSq'}.
            For more information see scipy.optimize module.
        """
        bnds = np.vstack(([(0., 10.)]*(self.prod_num + np.prod(self.tau_weights.shape)),
                          [(0., 1.)]*self.inj_num*self.prod_num))
        if 'pwf' in self.crm_type:
            bnds = np.vstack((bnds, [(0., 10.)]*np.prod(self.J_pwf.shape)))
        if dtype == 'N-M':
            result = scipy.optimize.minimize(self.objective_func, self.all_weights, method='Nelder-Mead')
        elif dtype == 'SLSQP':
            result = scipy.optimize.minimize(self.objective_func, self.all_weights, method='SLSQP', bounds=bnds)
        if dtype == 'BFGS':
            result = scipy.optimize.minimize(self.objective_func, self.all_weights, method='BFGS')#, bounds=bnds)
        elif dtype == 'DE':
            result = scipy.optimize.differential_evolution(self.objective_func, bnds)
        elif dtype == 'anneal':
            result = scipy.optimize.dual_annealing(self.objective_func, bnds)
        elif dtype == 'LSq':
            bnds = (bnds[:, 0], bnds[:, 1])
            result = scipy.optimize.least_squares(self.residuals, x0=self.all_weights, bounds=bnds)
        return result

    def predict(self, t_0, t_n):
        """Forward function to forecast rates."""
        return np.array([self.liquid_rate_calc(t_0, t_curr) for t_curr in range(t_0, t_n+1)])