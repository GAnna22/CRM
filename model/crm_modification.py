import numpy as np
import scipy.optimize
from sklearn.metrics import r2_score

class CRM_general:
    def __init__(self, t_0, t_n, dt, injection_rates, production_rates, dates, gtm_idx=None, P=None, method='bettas',
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
        dates: array of dates, shape (Nt, 1)
        gtm_idx: id of dates of GTMs in array 'dates' + id of the last element in array 'dates'
        P: number of the well on which GTMs are conducted
        method: str, {'bettas', 'pseudo_injectors'}
                - 'bettas' - predict effect of the GTM using bettas
                - 'pseudo_injectors' - predict effect of the GTM using fictional wells
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
        self.gtm_idx = gtm_idx
        self.method = method
        self.P = P
        self.t_0 = t_0
        self.t_n = t_n
        self.dt = dt
        self.dt_cum = dt.cumsum()
        self.dates = dates
        self.production_rates = production_rates
        self.production_rates_for_plot = production_rates
        if (gtm_idx is not None) and (method == 'pseudo_injectors'):
            """Adding fictional wells"""
            new_gtm = gtm_idx.copy()
            new_gtm[-1] = new_gtm[-1] + 1
            for idx in range(len(new_gtm) - 1):
                new_well = np.zeros((production_rates.shape[0], 1))
                new_well[new_gtm[idx]: new_gtm[idx + 1]] = self.production_rates[new_gtm[idx]: new_gtm[idx + 1], P - 1].reshape(-1, 1)
                self.production_rates = np.hstack([self.production_rates, new_well])
            self.production_rates[new_gtm[0]:, P - 1] = 0
        
        self.prod_num = self.production_rates.shape[1]
        self.injection_rates = injection_rates[:, :, None].repeat(self.prod_num, axis=-1)
        self.inj_num = injection_rates.shape[1]
        self.mask = self.production_rates == 0.
        zero_rows, zero_cols = np.where(self.mask)
        
    
        if pwf is not None:
            self.dpwf = np.vstack((pwf[0], pwf[1:] - pwf[:-1]))
            self.dpwf[zero_rows, zero_cols] = 0.
            self.dpwf[zero_rows+1, zero_cols] = 0.
            
        self.injection_rates[zero_rows, :, zero_cols] = 0.
        self.crm_type = crm_type
        if crm_type[:4] in ['CRMT', 'CRMP', 'CRMIP']:
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
                    
                if self.gtm_idx is not None:
                    if self.method == 'bettas':
                        
                        self.betta = np.random.uniform(0., 10., (len(self.gtm_idx[:-1]), self.prod_num))
                        self.all_weights = np.hstack((self.all_weights, self.betta.flatten()))

                        self.sigma = np.random.uniform(0., 10., (len(self.gtm_idx[:-1]), self.prod_num))
                        self.all_weights = np.hstack((self.all_weights, self.sigma.flatten()))
                                        
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
                    
                if self.gtm_idx is not None:
                    if self.method == 'bettas':
                        self.betta = all_weights[self.prod_num*(1+2*self.inj_num):-self.prod_num*
                                                 len(self.gtm_idx[:-1])].reshape(len(self.gtm_idx[:-1]), self.prod_num)
                        self.sigma = all_weights[-self.prod_num*len(self.gtm_idx[:-1]):].reshape(len(self.gtm_idx[:-1]), self.prod_num)
                    
    def CRMIPT(self, t_0, t_curr, GTM=False, gtm_idx=None):
        """Function to optimize without accounting for pressure drops."""
        if GTM:
            if self.method == 'bettas':
                conv_inj = 0.
                """Recalculation of lambdas through bettas."""
                lambda_weights = (self.lambda_weights + self.lambda_weights[:, self.P - 1].reshape(-1, 1)*
                                    self.betta[gtm_idx - 1, :].reshape(1, -1)*
                                    (1 - self.sigma[gtm_idx - 1, :].reshape(1, -1)))
                lambda_weights /= lambda_weights.sum(1)[:, None]
                for t_m in range(t_0 + 1, t_curr + 1):
                    exp_term = ((1 - np.exp(-self.dt[t_m - t_0]/self.tau_weights))*
                                 np.exp((self.dt_cum[t_m - t_0] - self.dt_cum[t_curr - t_0])/self.tau_weights))
                    conv_inj += (exp_term*self.injection_rates[t_m]*(1. + np.sum(self.mask[t_m]*lambda_weights, axis=1,
                                                                                 keepdims=True))
                                )

                return (self.production_rates[t_0]*np.exp(-(self.dt_cum[t_curr - t_0]-self.dt_cum[t_0 - t_0])/self.tau_prime) +
                        (lambda_weights*conv_inj).sum(0))
            
        else:
            conv_inj = 0.
            for t_m in range(t_0+1, t_curr+1):
                exp_term = ((1 - np.exp(-self.dt[t_m]/self.tau_weights))*
                             np.exp((self.dt_cum[t_m] - self.dt_cum[t_curr])/self.tau_weights))
                conv_inj += (exp_term*self.injection_rates[t_m]*(1. + np.sum(self.mask[t_m]*self.lambda_weights, axis=1,
                                                                             keepdims=True))
                            )
            return (self.production_rates[t_0]*np.exp(-(self.dt_cum[t_curr]-self.dt_cum[t_0])/self.tau_prime) +
                    (self.lambda_weights*conv_inj).sum(0))

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

    def residuals(self, all_weights, first_interval=True):
        """Residuals between predicitons and historical data."""
        if self.gtm_idx is not None and self.method == 'bettas':
            self.init_weights(all_weights)
            preds = np.array([self.liquid_rate_calc(self.t_0, t_curr) for t_curr in range(self.t_0, self.gtm_idx[0])])
            for i, gtm in enumerate(self.gtm_idx[:-1]):
                preds = np.vstack([preds, np.array([self.liquid_rate_calc(self.gtm_idx[i], t_curr, GTM=True, gtm_idx=i + 1) for t_curr in range(self.gtm_idx[i], self.gtm_idx[i + 1])])])
            return (preds - self.production_rates).flatten().astype('float64')
        
        else:
            self.init_weights(all_weights)
            preds = np.array([self.liquid_rate_calc(self.t_0, t_curr) for t_curr in range(self.t_0, self.t_n+1)])
            return (preds - self.production_rates[self.t_0:self.t_n+1]).flatten().astype('float64')

    def objective_func(self, all_weights):
        """MAE objective function to be minimized."""
        self.init_weights(all_weights)
        preds = np.array([self.liquid_rate_calc(self.t_0, t_curr) for t_curr in range(self.t_0, self.t_n+1)])
        self.error_func.append(
            ((preds - self.production_rates[self.t_0:self.t_n+1])**2).sum()/np.prod(preds.shape)
        ) #+ (self.mu_weights*(self.lambda_weights.sum(1) - 1)).sum()
        return self.error_func[-1]

    def optimize(self, dtype='LSq', verbose=False):
        """Optimization function. Constrained and unconstrained minimization.

        Parameters
        ----------
        dtype : str, {'N-M', 'SLSQP', 'BFGS', 'DE', 'anneal', 'LSq'}.
            For more information see scipy.optimize module.
        """
        bnds = np.vstack(([(0., 10.)]*(self.prod_num + np.prod(self.tau_weights.shape)),
                          [(0., 1.)]*self.inj_num*self.prod_num))
        if self.gtm_idx is not None:
            if self.method == 'bettas':
                bnds = np.vstack((bnds, [(0., 10.)]*2*self.prod_num*len(self.gtm_idx[:-1])))
        if 'pwf' in self.crm_type:
            bnds = np.vstack((bnds, [(0., 10.)]*np.prod(self.J_pwf.shape)))
        if dtype == 'N-M':
            result = scipy.optimize.minimize(self.objective_func, self.all_weights, method='Nelder-Mead')
        elif dtype == 'SLSQP':
            result = scipy.optimize.minimize(self.objective_func, self.all_weights, method='SLSQP', bounds=bnds)
        elif dtype == 'BFGS':
            result = scipy.optimize.minimize(self.objective_func, self.all_weights, method='BFGS')#, bounds=bnds)
        elif dtype == 'DE':
            result = scipy.optimize.differential_evolution(self.objective_func, bnds)
        elif dtype == 'anneal':
            result = scipy.optimize.dual_annealing(self.objective_func, bnds)
        elif dtype == 'LSq':
            bnds = (bnds[:, 0], bnds[:, 1])
            result = scipy.optimize.least_squares(self.residuals, x0=self.all_weights, bounds=bnds)
        if verbose:
            return result

    def predict(self, t_0, t_n):
        """Forward function to forecast rates."""
        if self.gtm_idx is not None and self.method == 'bettas':
            preds = np.array([self.liquid_rate_calc(self.t_0, t_curr) for t_curr in range(self.t_0, self.gtm_idx[0])])
            for i, gtm in enumerate(self.gtm_idx[:-1]):
                preds = np.vstack([preds, np.array([self.liquid_rate_calc(self.gtm_idx[i], t_curr, GTM=True, gtm_idx=i + 1) for t_curr in range(self.gtm_idx[i], self.gtm_idx[i + 1])])])
            return preds
        
        elif self.method == 'pseudo_injectors':
            predict = np.array([self.liquid_rate_calc(t_0, t_curr) for t_curr in range(t_0, t_n+1)])
            final_prediction = predict[:, :-len(self.gtm_idx) + 1]
            for i in range(1, len(gtm_idx)):
                final_prediction[:, self.P - 1] += predict[:, -i]
            return final_prediction
        
        else:
            return np.array([self.liquid_rate_calc(t_0, t_curr) for t_curr in range(t_0, t_n+1)])
    
    def plot_results(self, height=8, width=15):
        """ Plotting results"""
        
        if self.method == 'pseudo_injectors':
            num = self.prod_num - len(self.gtm_idx) + 1
        else:
            num = self.prod_num
        
        fig, ax = plt.subplots(num, 1, figsize=(width, height*self.prod_num))
        prediction = self.predict(self.t_0, self.production_rates.shape[0] - 1)
        
        for i in range(num):
            if self.gtm_idx is not None and self.method == 'bettas':
                r2 = r2_score(self.production_rates_for_plot[:, i], prediction[:, i])
                ax[i].plot(self.dates.reshape(-1)[:], prediction[:, i], '-or', label='Model prediction, $R^2$ = ' + str(round(r2, 3)))
            else:
                r2_train = r2_score(self.production_rates_for_plot[:self.t_n + 1, i], prediction[:self.t_n + 1, i])
                r2_test = r2_score(self.production_rates_for_plot[self.t_n + 1:, i], prediction[self.t_n + 1:, i])
                ax[i].plot(self.dates.reshape(-1)[:self.t_n + 1], prediction[:self.t_n + 1, i], '-or', label='Model prediction (train), $R^2$ = ' + str(round(r2_train, 3)))
                ax[i].plot(self.dates.reshape(-1)[self.t_n + 1:], prediction[self.t_n + 1:, i], '-og', label='Model prediction (test), $R^2$ = ' + str(round(r2_test, 3)))
                ax[i].plot(2*[self.dates[self.t_n + 1]], [min(self.production_rates_for_plot[:, i]), max(self.production_rates_for_plot[:, i])], '--')
            ax[i].plot(self.dates.reshape(-1)[:], self.production_rates_for_plot[:, i], '-ob', label='Actual rates')
            ax[i].set_title('P' + str(i), fontsize=15)
            ax[i].set_xlabel('Data', fontsize=15)
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].set_ylabel('Liquid Rate, sm3/day', fontsize=15)
            ax[i].grid()
            ax[i].legend(fontsize=15)