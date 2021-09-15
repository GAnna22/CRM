import numpy as np
import scipy.optimize
import pandas as pd
import torch

class CRM:
    def __init__(self, t_0, t_n, timestamp, dates, injection_rates, liquid_rates,
                 pwf=None, bhp_term=True, well_distances=None, radius_for_accounting=None,
                 gtm_info=None, aquifer=False, lambda_matbal=False):
        self.t_0 = t_0
        self.t_n = t_n
        self.timestamp = timestamp
        self.t_max = len(dates)
        self.dates = dates
        self.dt = np.hstack(([1.], pd.to_timedelta(dates.values[1:] -
                                                   dates.values[:-1]).days/timestamp))
        self.dt_cum = self.dt.cumsum(0)
        self.well_distances = well_distances
        self.radius_for_accounting = radius_for_accounting
        self.bhp_term = bhp_term
        self.gtm_info = gtm_info
        self.aquifer = aquifer
        self.lambda_matbal = lambda_matbal
        self.define_lambdas()
        self._prime_rates = liquid_rates[t_0]
        self._prime_rates[np.isclose(self._prime_rates.round(0), 0.)] = self._prime_rates.max()
        self._prime_rates = self._prime_rates[None].repeat(self.t_max, axis=0)

        self.loss_error = []
        self.predict_liq = self._forward
        self.preprocessing(injection_rates, liquid_rates, pwf)

        self.MSE = lambda preds, targets: ((preds - targets)**2).mean(0)
        self.MAE = lambda preds, targets: (abs(preds - targets)).mean(0)
        self.R2 = lambda preds, targets: 1 - ((preds - targets)**2).sum(0)/((targets - targets.mean(0))**2).sum(0)

    def init_dates(self, dates, timestamp):
        self.timestamp = timestamp
        self.t_max = len(dates)
        self.dates = dates
        self.dt = np.hstack(([1.], pd.to_timedelta(dates.values[1:] -
                                                   dates.values[:-1]).days/timestamp))
        self.dt_cum = self.dt.cumsum(0)

    def init_prime_rates(self, liquid_prime):
        self._prime_rates = liquid_prime
        self._prime_rates[np.isclose(self._prime_rates.round(0), 0.)] = self._prime_rates.max()
        self._prime_rates = self._prime_rates[None].repeat(self.t_max, axis=0)

    @property
    def prime_rates(self):
        return self._prime_rates

#     def prime_rates(self, t_0, t_n):
#         self._prime_rates = self.liquid_rates[t_0].copy()
#         self._prime_rates[np.isclose(self._prime_rates.round(0), 0.)] = self._prime_rates.max()
#         self._prime_rates = self._prime_rates[None].repeat(self.t_max, axis=0)

    def pressure_processing(self, pwf):
        if self.bhp_term:
            self.pwf = pwf
#             pwf[zero_rows, zero_cols] = pwf.max(0, keepdims=True).repeat(self.t_max, 0)[zero_rows, zero_cols]
            self.dpwf = np.vstack((np.zeros_like(pwf[0]), pwf[1:] - pwf[:-1]))
            self.dpwf[(self.zero_rows, self.zero_rows_1), self.zero_cols] = 0.
            self.dpwf2 = pwf - pwf[self.t_0]
            self.dpwf2[(self.zero_rows, self.zero_rows_1), self.zero_cols] = 0.
            self.dpwf_max = np.abs(self.dpwf).max(0)
            self.pwf = (1. - self.shut_mask)*self.pwf
            self.pwf_max = np.abs(self.pwf).max(0)
            self.shut_mask2 = np.zeros((self.t_max, self.prod_num))
            for item in np.unique(self.zero_cols):
                inds1 = np.setdiff1d(self.zero_rows[self.zero_cols == item],
                                     self.zero_rows_1[self.zero_cols == item])
                inds2 = np.setdiff1d(self.zero_rows_1[self.zero_cols == item],
                                     self.zero_rows[self.zero_cols == item])
                self.shut_mask2[inds1, item] = 1.
                self.shut_mask2[inds2, item] = -1.

    def preprocessing(self, injection_rates, liquid_rates, pwf):
        self.liquid_rates = liquid_rates
        self.liq_mean_vals = liquid_rates[self.t_0:self.t_n+1].mean(0)
#         self.loss_weights = ((liquid_rates[self.t_0:self.t_n+1].max(0) -
#                               liquid_rates[self.t_0:self.t_n+1].min(0)).max()/
#                              (liquid_rates[self.t_0:self.t_n+1].max(0) -
#                               liquid_rates[self.t_0:self.t_n+1].min(0)))
#         self.loss_weights = self.loss_weights/self.loss_weights.sum()
        self.t_max, self.prod_num = self.liquid_rates.shape
        self.inj_num = injection_rates.shape[1]
        if (self.well_distances[0] is None) or (self.radius_for_accounting[0] is None):
            self.well_distances[0] = np.ones((self.inj_num, self.prod_num))
            self.radius_for_accounting[0] = 1.
        self.prod_inj_distance_mask = np.sign(pd.DataFrame(self.well_distances[0]).where(
            pd.DataFrame(self.well_distances[0]) <= self.radius_for_accounting[0], 0.).values)
        self.active_inj_inds = np.setdiff1d(np.arange(self.inj_num),
                                            np.where(self.prod_inj_distance_mask.sum(1) == 0.)[0])
        self.inj_num = len(self.active_inj_inds)
        self.injection_rates = injection_rates[:, self.active_inj_inds, None].repeat(self.prod_num, axis=-1)
        self.inj_mean_val = self.injection_rates[self.t_0:self.t_n+1, :, 0].mean()
        self.inj_min_vals, self.inj_max_vals = (self.injection_rates[self.t_0:self.t_n+1, :, 0].min(0),
                                                self.injection_rates[self.t_0:self.t_n+1, :, 0].max(0))
        self.prod_inj_distance_mask = self.prod_inj_distance_mask[self.active_inj_inds, :]
        self.shut_mask = np.isclose(self.liquid_rates.round(0), 0.)
        self.zero_rows, self.zero_cols = np.where(self.shut_mask)
        zero_rows_1 = self.zero_rows[:] + 1
        self.zero_rows_1 = np.where(zero_rows_1 > self.t_max-1, self.t_max-1, zero_rows_1)
        zero_rows_inj = np.setdiff1d(self.zero_rows,
                                     np.where((self.shut_mask.sum(-1) == self.prod_num))[0],
                                     assume_unique=True)
        zero_cols_inj = self.zero_cols[[ind for ind, elem in enumerate(self.zero_rows)
                                        if elem in zero_rows_inj]]
        self.injection_rates[zero_rows_inj, :, zero_cols_inj] = 0.
#         self.injection_rates[self.zero_rows, :, self.zero_cols] = 0.
        self.shut_mask3 = np.zeros((self.t_max, self.inj_num, self.prod_num))
        self.shut_mask3[self.zero_rows, :, self.zero_cols] = 1.
        self.shut_mask4 = self.shut_mask3.copy()
        self.shut_mask4[np.where(self.shut_mask3.sum(-1) == self.prod_num)[0]] = 0.
        self.pressure_processing(pwf)
        if self.gtm_info is not None:
            self.gtm_info = np.array(self.gtm_info)
            self.gtm_info = self.gtm_info[np.argsort(self.gtm_info, axis=0)[:, 0]]
            self.gtm_info = np.hstack((np.vstack((np.vstack((self.gtm_info[:-1, 0], self.gtm_info[1:, 0])).T,
                                                  [[self.gtm_info[-1, 0], self.t_max]])),
                                       self.gtm_info[:, None, 1]))
            self.gtm_info = self.gtm_info[self.gtm_info[:, 0] >= self.t_0]
            self.gtm_info = self.gtm_info[self.gtm_info[:, 0] <= self.t_n]
            self.gtm_info[np.where(self.gtm_info[:, 1] >= self.t_n)[0], 1] = self.t_max
            self.gtm_info = self.gtm_info[(self.gtm_info[:, 0] < self.gtm_info[:, 1])]
            self.num_of_gtm = self.gtm_info.shape[0]
            self.gtm_mask = np.zeros((self.t_max, self.inj_num, self.prod_num))
            for t1, t2, well_ind in self.gtm_info:
                self.gtm_mask[t1:t2, :, well_ind] = 1
            if self.num_of_gtm == 0:
                self.gtm_info = None

    def set_boundaries(self):
        return

    def init_weights(self):
        return

    def define_lambdas(self):
        if self.lambda_matbal:
            self.lambdas = lambda weights: weights/weights.sum(-1, keepdims=True)
        else:
            self.lambdas = lambda weights: weights

    def reset_train_frame(self, t_0, t_n):
        self.t_0 = t_0
        self.t_n = t_n

    def reset_loss_error(self):
        self.loss_error = []

    def _forward(self, t_0, t_n):
        self.prime_term = (self.liquid_rates[t_0]*self.lambda_prime*
                           np.exp(-(self.dt_cum[t_0:t_n+1]-self.dt_cum[t_0])[:, None]/self.tau_prime))
        self.exp_term = ((1. - np.exp(-self.dt[t_0+1:t_n+1, None, None]/self.tau_weights))*
                         np.exp(-(np.flip(self.dt_cum[t_0+1:t_n+1])-self.dt_cum[t_0+1])[:, None, None]/
                                self.tau_weights)
                        )
        self.masked_lambdas = np.zeros_like(self.injection_rates) + 1.
        if self.gtm_info is not None:
            for ind, (t1, t2, well_ind) in enumerate(self.gtm_info):
                self.masked_lambdas[t1:t2] = ((1 + self.gtm_mask[t1:t2]*self.betas[ind, :, None] +
                                              (self.gtm_mask[t1:t2] - 1.)*self.betas[ind, :, None]*
                                              (self.prod_inj_distance_mask*
                                               self.lambdas(self.lambda_weights))[None, :, well_ind, None])*
                                              self.prod_inj_distance_mask)
        self.masked_inj = ((self.masked_lambdas[t_0+1:t_n+1]*
                            (1 + np.sum(self.shut_mask[t_0+1:t_n+1, None, :]*
                                        (self.lambdas(self.lambda_weights)*self.masked_lambdas[t_0+1:t_n+1]),
                                        axis=2, keepdims=True))
                           )*self.injection_rates[t_0+1:t_n+1])
        self.conv_inj = np.vstack((np.zeros((1, self.inj_num, self.prod_num)),
                                   [(self.exp_term[-(i+1):]*self.masked_inj[:i+1]).sum(0)
                                    for i in range(t_n-t_0)]
                                  ))
        if self.bhp_term:
            self.conv_pwf = np.vstack((np.zeros((1, self.prod_num)),
                                       (self.tau_weights*self.J_pwf*
                                        [(self.exp_term[-(i+1):]*
                                          (self.dpwf[t_0+1:t_n+1, None, :]/
                                           self.dt[t_0+1:t_n+1, None, None])[:i+1]).sum(0)
                                         for i in range(t_n-t_0)]).sum(1)
                                      ))
        else:
            self.conv_pwf = 0.
        return self.prime_term + (self.lambdas(self.lambda_weights)*self.conv_inj).sum(1) - self.conv_pwf

#     def _oil_empirical(self, t_0, t_n, liquid_rates):
#         masked_inj = (self.injection_rates[t_0:t_n+1]*
#                       (1 + np.sum(self.shut_mask[t_0:t_n+1, None, :]*
#                                   self.lambdas(self.lambda_weights*self.masked_lambdas[t_0:t_n+1]),
#                                   axis=2, keepdims=True))
#                      )
#         self.cum_injection = ((self.lambdas(self.lambda_weights*self.masked_lambdas[t_0:t_n+1])*
#                                masked_inj*self.dt[t_0:t_n+1, None, None]).sum(1).cumsum(0)
#                              )
#         return liquid_rates/(1 + np.exp(-self.a_weights)*self.cum_injection**self.b_weights)

#     def _oil_frac(self, t_0, t_n):
#         masked_inj = (self.injection_rates[t_0:t_n+1]*
#                       (1 + np.sum(self.shut_mask[t_0:t_n+1, None, :]*
#                                   self.lambdas(self.lambda_weights*self.masked_lambdas[t_0:t_n+1]),
#                                   axis=2, keepdims=True))
#                      )
#         cum_injection = ((self.lambdas(self.lambda_weights*self.masked_lambdas[t_0:t_n+1])*
#                           masked_inj*self.dt[t_0:t_n+1, None, None]).sum(1).cumsum(0)
#                         )
#         return self.b_weights*np.log(cum_injection) - self.a_weights

#     def _oil_empirical(self, t_0, t_n, liquid_rates):
#         self.cum_injection = ((self.lambdas_vs_time*self.injection_rates[t_0:t_n+1]*
#                                self.dt[t_0:t_n+1, None, None]).sum(1).cumsum(0)
#                              )
#         return liquid_rates/(1 + np.exp(-self.a_weights)*self.cum_injection**self.b_weights)

    def add_ffm(self, t_0_oil, t_n_oil, oil_rates=None, water_rates=None, dtype='Empirical', ff_weights=None):
        self.ffm_type = dtype
        self.oil_rates = oil_rates
        self.water_rates = water_rates
        self.t_0_oil, self.t_n_oil = t_0_oil, t_n_oil
        if dtype == 'Empirical':
            self.predict_oil = self._oil_empirical
            self.predict_water = self._water_empirical
            self.frac_flow_model = self._frac_empirical
        elif dtype == 'BL':
            self.sat_start = None
            self.predict_oil = self._oil_bl
            self.predict_water = self._water_bl
            self.frac_flow_model = self._frac_bl
        self.set_ff_boundaries()
        self.init_ff_weights(ff_weights)

    def set_ff_boundaries(self, bnds_ff_dict={'Empirical': {'a_weights': (0., 50.), 'b_weights': (0.1, 5.)},
                                              'BL': {'Mo_weights': (0., 5.), 'Vp_weights': (1e-5, 50.),
                                                     'swr_weights': (0., 0.4), 'sor_weights': (0., 0.4),
                                                     'm_weights': (0., 20.), 'n_weights': (0., 5.)}
                                             }):
        self.bnds_ff_dict = bnds_ff_dict
        self.ff_bnds = []
        for k, v in self.bnds_ff_dict[self.ffm_type].items():
            self.ff_bnds.append([v]*self.prod_num)
        self.ff_bnds = np.vstack((self.ff_bnds))

    def init_ff_weights(self, ff_weights=None):
        if ff_weights is None:
            self.ff_weights = []
            for k, v in self.bnds_ff_dict[self.ffm_type].items():
                setattr(self, k, np.random.uniform(v[0], v[1], self.prod_num))
                self.ff_weights = np.hstack((self.ff_weights, getattr(self, k)))
        else:
            self.ff_weights = ff_weights
            for i, k in enumerate(list(self.bnds_ff_dict[self.ffm_type].keys())):
                setattr(self, k, ff_weights[i*self.prod_num:(i+1)*self.prod_num])
        if self.ffm_type == 'BL':
            if self.sat_start is not None:
                self.swr_weights = self.sat_start
        self.jac_sparsity = np.hstack((self.jac_sparsity,
                                       [np.ones(len(list(self.bnds_ff_dict[self.ffm_type].keys()))*self.prod_num)]*
                                       self.prod_num*(self.t_n+1-self.t_0)
                                      ))

    def _oil_empirical(self, t_0, t_n, liquid_rates):
        oil_frac = self.frac_flow_model(t_0, t_n)
        return liquid_rates[1:]/(1 + np.exp(oil_frac))

    def _water_empirical(self, t_0, t_n, liquid_rates):
        oil_frac = self.frac_flow_model(t_0, t_n)
        return (1. - 1/(1 + np.exp(oil_frac)))*liquid_rates[1:]

    def _frac_empirical(self, t_0, t_n):
        self.cum_injection = ((self.lambdas_vs_time[t_0+1:t_n+1]*self.injection_rates[t_0+1:t_n+1]*
                               self.dt[t_0+1:t_n+1, None, None]*self.timestamp).cumsum(0).sum(1)
                             )
        return self.b_weights*np.log(self.cum_injection) - self.a_weights

    def _oil_bl(self, t_0, t_n, liquid_rates):
        oil_frac = self.frac_flow_model(t_0, t_n)
        return oil_frac[1:]*liquid_rates

    def _water_bl(self, t_0, t_n, liquid_rates):
        oil_frac = self.frac_flow_model(t_0, t_n)
        return (1. - oil_frac[1:])*liquid_rates

    def set_sat_start(self, swr=None):
        self.sat_start = swr

    def _frac_bl(self, t_0, t_n):
        def water_frac(t_k):
            if t_k <= t_n:
                curr_inj = (self.lambdas_vs_time[t_k]*self.injection_rates[t_k]).sum(0)
                self.sw.append(self.sw[-1] + ((curr_inj - self.water_rates[t_k])/
                                              (self.Vp_weights*10**5)
                                             )
                              )
                sD = ((self.sw[-1] - self.swr_weights)**2)**0.5/(1. - self.sor_weights - self.swr_weights)
                self.fw_bl.append(1./(1 + (1. - sD)**self.m_weights/(self.Mo_weights*sD**self.n_weights)))
                return water_frac(t_k+1)
            else:
                return
        self.sw = [self.swr_weights]
        self.fw_bl = [self.water_rates[t_0]/self.liquid_rates[t_0]]
        _ = water_frac(t_0)
        return 1. - np.array(self.fw_bl)

    def define_bounds(self, fluids, constraints):
        if 'liq' in fluids:
            x0_weights = self.all_weights
            bnds = self.bnds
            jac_sparsity = self.jac_sparsity[:, :len(x0_weights)]
            if 'oil' in fluids:
                x0_weights = np.hstack((x0_weights, self.ff_weights))
                bnds = np.vstack((bnds, self.ff_bnds))
                jac_sparsity = self.jac_sparsity[:, :len(x0_weights)]
                jac_sparsity = np.vstack((jac_sparsity[1:], jac_sparsity[1:]))
        else:
            x0_weights = self.ff_weights
            bnds = self.ff_bnds
            jac_sparsity = self.jac_sparsity[1:, -len(x0_weights):]
        if not constraints:
            bnds = None
        return x0_weights, bnds, jac_sparsity

    def loss_liq(self, params, loss_fn):
        self.init_weights(params)
        liq_preds = self.predict_liq(self.t_0, self.t_n)
        outp = loss_fn(liq_preds, self.liquid_rates[self.t_0:self.t_n+1])
        return outp

    def residuals_liq(self, params):
        self.init_weights(params)
        liq_preds = self.predict_liq(self.t_0, self.t_n)
#         mask = np.where(1 - self.shut_mask[self.t_0:self.t_n+1].flatten())[0]
        outp = (liq_preds - self.liquid_rates[self.t_0:self.t_n+1]).flatten()#[mask]
        return outp

#     def loss_oil(self, params, loss_fn):
#         self.init_ff_weights(params[-len(self.ff_weights):])
#         oil_frac_preds = self.frac_flow_model(self.t_0, self.t_n)
#         return loss_fn(np.log(oil_frac_preds**(-1) - 1.), np.log(self.liquid_rates[self.t_0:self.t_n+1]/
#                                               self.oil_rates[self.t_0:self.t_n+1] - 1.))
    def loss_oil(self, params, loss_fn):
        self.init_ff_weights(params[-len(self.ff_weights):])
        oil_frac_preds = self.frac_flow_model(self.t_0_oil, self.t_n_oil)
        if self.ffm_type == 'Empirical':
            return loss_fn(self.liquid_rates[self.t_0_oil+1:self.t_n_oil+1]/(1+np.exp(oil_frac_preds)),
                           self.oil_rates[self.t_0_oil+1:self.t_n_oil+1])
        elif self.ffm_type == 'BL':
            return loss_fn(oil_frac_preds[1:]*self.liquid_rates[self.t_0_oil:self.t_n_oil+1],
                           self.oil_rates[self.t_0_oil:self.t_n_oil+1])

#     def residuals_oil(self, params):
#         self.init_ff_weights(params[-len(self.ff_weights):])
#         oil_frac_preds = self.frac_flow_model(self.t_0, self.t_n)
#         return (np.log(oil_frac_preds**(-1) - 1.) -
#                 np.log(self.liquid_rates[self.t_0:self.t_n+1]/
#                        self.oil_rates[self.t_0:self.t_n+1] - 1.)).flatten()
    def residuals_oil(self, params):
        self.init_ff_weights(params[-len(self.ff_weights):])
        oil_frac_preds = self.frac_flow_model(self.t_0_oil, self.t_n_oil)
        if self.ffm_type == 'Empirical':
            return (self.liquid_rates[self.t_0_oil+1:self.t_n_oil+1]/(np.exp(oil_frac_preds) + 1) -
                    self.oil_rates[self.t_0_oil+1:self.t_n_oil+1]).flatten()
        elif self.ffm_type == 'BL':
            return (oil_frac_preds[1:]*self.liquid_rates[self.t_0_oil:self.t_n_oil+1] -
                    self.oil_rates[self.t_0_oil:self.t_n_oil+1]).flatten()

    def optimize(self, opt_type='LSq', loss_type='MAE', fluids='liq',
                 constraints=True, train_frame=None):
        if train_frame is not None:
            t_0, t_n = train_frame
            self.reset_train_frame(t_0, t_n)

        mse_fn = lambda preds, targets: (((preds - targets)**2).mean(0)).sum()
        mae_fn = lambda preds, targets: (np.abs(preds - targets).mean(0)).sum()
        loss_fn = mse_fn if loss_type=='MSE' else mae_fn

        if ('liq' in fluids) and ('oil' not in fluids):
            def loss(params):
                outp = self.loss_liq(params, loss_fn)
                self.loss_error.append(outp)
                return outp
            def residuals(params):
                outp = self.residuals_liq(params)
                self.loss_error.append(loss_fn(outp, 0.))
                return outp
        elif ('liq' in fluids) and ('oil' in fluids):
            def loss(params):
                outp = self.loss_liq(params, loss_fn) + self.loss_oil(params, loss_fn)
                self.loss_error.append(outp)
                return outp
            def residuals(params):
                outp = np.hstack((self.residuals_liq(params),
                                  self.residuals_oil(params)))
                self.loss_error.append(loss_fn(outp, 0.))
                return outp
        else:
            def loss(params):
                outp = self.loss_oil(params, loss_fn)
                self.loss_error.append(outp)
                return outp
            def residuals(params):
                outp = self.residuals_oil(params)
                self.loss_error.append(loss_fn(outp, 0.))
                return outp

        x0_weights, bnds, jac_sparsity = self.define_bounds(fluids, constraints)

        if opt_type in ['Nelder-Mead', 'BFGS']:
            result = scipy.optimize.minimize(loss, x0_weights, method=opt_type)
        if opt_type in ['SLSQP', 'COBYLA', 'trust-constr']:
            if opt_type == 'COBYLA':
                bnds = None
            result = scipy.optimize.minimize(loss, x0_weights, method=opt_type, bounds=bnds)
        if opt_type == 'DE':
            result = scipy.optimize.differential_evolution(loss, bnds)
        if opt_type == 'Anneal':
            result = scipy.optimize.dual_annealing(loss, bnds, x0=x0_weights)
        if opt_type == 'LSq':
            if constraints:
                bnds = (bnds[:, 0], bnds[:, 1])
            result = scipy.optimize.least_squares(residuals, x0=x0_weights, bounds=bnds,
#                                                   jac_sparsity=jac_sparsity,
                                                  ftol=1e-06, xtol=1e-06, gtol=1e-06) #, x_scale='jac')
#         if opt_type == 'root':
#             result = scipy.optimize.root(residuals, x0_weights, method='hybr')
        return result

    def to_numpy(self):
#         self.loss_weights = torch.tensor(self.loss_weights, dtype=torch.float)
        self.dt = torch.tensor(self.dt, dtype=torch.float)
        self.dt_cum = torch.tensor(self.dt_cum, dtype=torch.float)
        self.injection_rates = torch.tensor(self.injection_rates, dtype=torch.float)
        self.liquid_rates = torch.tensor(self.liquid_rates, dtype=torch.float)
        self.shut_mask = torch.tensor(self.shut_mask, dtype=torch.float)

        self.lambda_prime = torch.tensor(self.lambda_prime, dtype=torch.float, requires_grad=True)
        self.tau_prime = torch.tensor(self.tau_prime, dtype=torch.float, requires_grad=True)
        self.lambda_weights = torch.tensor(self.lambda_weights, dtype=torch.float, requires_grad=True)
        self.tau_weights = torch.tensor(self.tau_weights, dtype=torch.float, requires_grad=True)

        if self.bhp_term:
            self.dpwf = torch.tensor(self.dpwf, dtype=torch.float)
            self.dpwf2 = torch.tensor(self.dpwf2, dtype=torch.float)
            self.J_pwf = torch.tensor(self.J_pwf, dtype=torch.float, requires_grad=True)
        if self.gtm_info is not None:
            self.gtm_mask = torch.tensor(self.gtm_mask, dtype=torch.float)
        self.predict_liq = self._forward_torch

    def define_lambdas_torch(self):
        if self.lambda_matbal:
            self.lambdas = lambda weights: torch.exp(-weights)/torch.exp(-weights).sum(1, keepdims=True)
        else:
            self.lambdas = lambda weights: weights

    def to_torch(self):
#         self.loss_weights = torch.tensor(self.loss_weights, dtype=torch.float)
        self.dt = torch.tensor(self.dt, dtype=torch.float)
        self.dt_cum = torch.tensor(self.dt_cum, dtype=torch.float)
        self.injection_rates = torch.tensor(self.injection_rates, dtype=torch.float)
        self.liquid_rates = torch.tensor(self.liquid_rates, dtype=torch.float)
        self.shut_mask = torch.tensor(self.shut_mask, dtype=torch.float)

        self.lambda_prime = torch.tensor(self.lambda_prime, dtype=torch.float, requires_grad=True)
        self.tau_prime = torch.tensor(self.tau_prime, dtype=torch.float, requires_grad=True)
        self.lambda_weights = torch.tensor(self.lambda_weights, dtype=torch.float, requires_grad=True)
        self.tau_weights = torch.tensor(self.tau_weights, dtype=torch.float, requires_grad=True)

        if self.bhp_term:
            self.dpwf = torch.tensor(self.dpwf, dtype=torch.float)
            self.dpwf2 = torch.tensor(self.dpwf2, dtype=torch.float)
            self.J_pwf = torch.tensor(self.J_pwf, dtype=torch.float, requires_grad=True)
        if self.gtm_info is not None:
            self.gtm_mask = torch.tensor(self.gtm_mask, dtype=torch.float)
            self.betas = torch.tensor(self.betas, dtype=torch.float, requires_grad=True)
        self.predict_liq = self._forward_torch

    def _forward_torch(self, t_0, t_n):
        self.prime_term = (self.liquid_rates[t_0]*self.lambda_prime*
                           torch.exp(-(self.dt_cum[t_0:t_n+1]-self.dt_cum[t_0])[:, None]/self.tau_prime))
        self.exp_term = ((1. - torch.exp(-self.dt[t_0+1:t_n+1, None, None]/self.tau_weights))*
                         torch.exp(-(torch.flip(self.dt_cum[t_0+1:t_n+1], dims=(0,))-
                                     self.dt_cum[t_0+1])[:, None, None]/
                                   self.tau_weights)
                        )
        self.masked_lambdas = torch.zeros_like(self.injection_rates) + 1.
        if self.gtm_info is not None:
            for ind, (t1, t2, well_ind) in enumerate(self.gtm_info):
                self.masked_lambdas[t1:t2] = (1 + self.gtm_mask[t1:t2]*self.betas[ind, :, None] +
                                              (self.gtm_mask[t1:t2] - 1.)*self.betas[ind, :, None]*
                                               self.lambda_weights[:, well_ind, None])
        self.masked_inj = (self.injection_rates[t_0+1:t_n+1]*
                           (1 + torch.sum(self.shut_mask[t_0+1:t_n+1, None, :]*
                                          self.lambdas(self.prod_inj_distance_mask*self.lambda_weights)*
                                                       self.masked_lambdas[t_0+1:t_n+1],
                                          dim=2, keepdims=True))
                          )
        self.conv_inj = torch.cat((torch.zeros((1, self.inj_num, self.prod_num)),
                                   torch.stack([(self.exp_term[-(i+1):]*self.masked_inj[:i+1]).sum(0)
                                                 for i in range(t_n-t_0)])
                                  ), dim=0)
        if self.bhp_term:
            self.conv_pwf = torch.cat((torch.zeros((1, self.prod_num)),
                                       (self.tau_weights*self.J_pwf*
                                        torch.stack([(self.exp_term[-(i+1):]*
                                                      (self.dpwf[t_0+1:t_n+1, None, :]/
                                                       self.dt[t_0+1:t_n+1, None, None])[:i+1]).sum(0)
                                                     for i in range(t_n-t_0)])).sum(1)
                                      ), dim=0)
        else:
            self.conv_pwf = 0.
        return (self.prime_term +
                (self.lambdas(self.lambda_weights*self.masked_lambdas[t_0:t_n+1])*self.conv_inj).sum(1) -
                self.conv_pwf)

    def define_loss_torch(self, fluids, loss_type):
        if loss_type == 'MSE':
            loss_fn = torch.nn.MSELoss()
        elif loss_type == 'MAE':
            loss_fn = torch.nn.L1Loss()
        elif loss_type == 'Huber':
            loss_fn = torch.nn.SmoothL1Loss()
        elif loss_type == 'LogCosh':
            loss_fn = lambda preds, targets: torch.log(torch.cosh(preds - targets)).mean()
        elif loss_type == 'KLDIV':
            loss_fn = lambda preds, targets: torch.nn.KLDivLoss(reduction='batchmean')(torch.log(preds), targets)
        if 'liq' in fluids:
            params = [self.lambda_prime, self.tau_prime, self.lambda_weights, self.tau_weights]
            if self.bhp_term:
                params.append(self.J_pwf)
            if 'oil' in fluids:
                params += [self.a_weights, self.b_weights]
        else:
            params = [self.a_weights, self.b_weights]
        return loss_fn, params

    def optimize_torch(self, opt_type='Adam', loss_type='MAE', fluids='liq',
                       constraints=True, train_frame=None):
        if train_frame:
            t_0, t_n = train_frame
            self.reset_train_frame(t_0, t_n)
        self.to_torch()
        self.define_lambdas_torch()
        self.loss_fn, self.params = self.define_loss_torch(fluids, loss_type)

class CRMP(CRM):
    def __init__(self, t_0, t_n, timestamp, dates, injection_rates, liquid_rates,
                 pwf=None, bhp_term=True, well_distances=None, radius_for_accounting=None,
                 gtm_info=None, aquifer=False, lambda_matbal=False, all_weights=None):
        super().__init__(t_0, t_n, timestamp, dates, injection_rates, liquid_rates,
                         pwf, bhp_term, well_distances, radius_for_accounting,
                         gtm_info, aquifer, lambda_matbal)
        zero_rows_inj = np.setdiff1d(self.zero_rows,
                                     np.where((self.shut_mask.sum(-1) == self.prod_num))[0],
                                     assume_unique=True)
        zero_cols_inj = self.zero_cols[[ind for ind, elem in enumerate(self.zero_rows) if elem in zero_rows_inj]]
#         self.injection_rates[zero_rows_inj, :, zero_cols_inj] = 0.
        self.injection_rates[self.zero_rows, :, self.zero_cols] = 0.
        self.set_boundaries()
        self.init_weights(all_weights)

    def set_boundaries(self, bnds_dict={'lambda_prime': (0., 1.), 'tau_prime': (0.0001, 20.),
                                        'lambda_weights': (0., 1.), 'tau_weights': (0.0001, 20.),
                                        'J_pwf': (0., 20.), 'betas': (0., 10.),
                                        'p_res_weights': (0., 5.), 'aquifer_weights': (-0.5, 0.5)
                                       }):
        self.bnds_dict = bnds_dict
        self.bnds = np.vstack(([self.bnds_dict['lambda_prime']]*self.prod_num,
                               [self.bnds_dict['tau_prime']]*self.prod_num,
                               [self.bnds_dict['lambda_weights']]*self.prod_num*self.inj_num,
                               [self.bnds_dict['tau_weights']]*self.prod_num))
        if self.bhp_term:
            self.bnds = np.vstack((self.bnds, [self.bnds_dict['J_pwf']]*self.prod_num))
        if self.gtm_info is not None:
            self.bnds = np.vstack((self.bnds, [self.bnds_dict['betas']]*
                                   self.num_of_gtm*self.prod_num))
#         if self.bhp_term:
#             self.bnds = np.vstack((self.bnds, [self.bnds_dict['p_res_weights']]*self.prod_num))
        if self.aquifer:
            self.bnds = np.vstack((self.bnds, [self.bnds_dict['aquifer_weights']]*self.prod_num))

    def init_weights(self, all_weights=None):
        if all_weights is None:
            self.lambda_prime = np.random.uniform(0.01, 1., self.prod_num)
            self.tau_prime = np.random.uniform(0.1, 1., self.prod_num)
            self.lambda_weights = np.random.uniform(0.01, 1., (self.inj_num, self.prod_num))
            self.lambda_weights /= np.sum(self.lambda_weights, axis=1, keepdims=True)
            self.tau_weights = np.random.uniform(0.1, 1., (1, self.prod_num))
            self.all_weights = np.hstack((self.lambda_prime, self.tau_prime,
                                          self.lambda_weights.flatten(), self.tau_weights.flatten()
                                         ))
            self.jac_sparsity = np.hstack((np.ones(2*self.prod_num),
                                           self.prod_inj_distance_mask.flatten(),
                                           np.ones(self.prod_num)
                                          ))
            if self.bhp_term:
                self.J_pwf = np.random.uniform(1., 5., (1, self.prod_num))
                self.all_weights = np.hstack((self.all_weights, self.J_pwf.flatten()))
                self.jac_sparsity = np.hstack((self.jac_sparsity, np.ones(self.prod_num)))
            if self.gtm_info is not None:
                self.betas = np.random.uniform(0., 0.1, (self.num_of_gtm, self.prod_num))
                self.all_weights = np.hstack((self.all_weights, self.betas.flatten()))
                self.jac_sparsity = np.hstack((self.jac_sparsity,
                                               np.ones(self.num_of_gtm*self.prod_num)))
#             if self.bhp_term:
#                 self.p_res_weights = np.random.uniform(0.5, 1.5, self.prod_num)
#                 self.all_weights = np.hstack((self.all_weights, self.p_res_weights))
#                 self.jac_sparsity = np.hstack((self.jac_sparsity, np.ones(self.prod_num)))
            if self.aquifer:
                self.aquifer_weights = np.random.uniform(-0.05, 0.05, self.prod_num)
                self.all_weights = np.hstack((self.all_weights, self.aquifer_weights))
                self.jac_sparsity = np.hstack((self.jac_sparsity, np.ones(self.prod_num)))
            self.jac_sparsity = self.jac_sparsity[None].repeat(self.prod_num*(self.t_n+1-self.t_0), 0)
        else:
            self.all_weights = all_weights
            self.lambda_prime = all_weights[:self.prod_num]
            self.tau_prime = all_weights[self.prod_num:2*self.prod_num]
            self.lambda_weights = all_weights[2*self.prod_num:self.prod_num*(2+self.inj_num)
                                             ].reshape(self.inj_num, self.prod_num)
            self.tau_weights = all_weights[self.prod_num*(2+self.inj_num):
                                           self.prod_num*(3+self.inj_num)].reshape(1, self.prod_num)
            if self.bhp_term:
                self.J_pwf = all_weights[self.prod_num*(3+self.inj_num):
                                         self.prod_num*(4+self.inj_num)].reshape(1, self.prod_num)
                if self.gtm_info is not None:
                    self.betas = all_weights[self.prod_num*(4+self.inj_num):
                                             self.prod_num*(4+self.inj_num)+
                                             self.num_of_gtm*self.prod_num
                                            ].reshape(self.num_of_gtm, self.prod_num)
#                     self.p_res_weights = all_weights[self.prod_num*(4+self.inj_num)+
#                                                      self.num_of_gtm*self.prod_num:
#                                                      self.prod_num*(4+self.inj_num)+
#                                                      self.prod_num*(self.num_of_gtm+1)]
#                 else:
#                     self.p_res_weights = all_weights[self.prod_num*(4+self.inj_num):
#                                                      self.prod_num*(5+self.inj_num)]
            else:
                if self.gtm_info is not None:
                    self.betas = all_weights[self.prod_num*(3+self.inj_num):
                                             self.prod_num*(3+self.inj_num)+
                                             self.num_of_gtm*self.prod_num
                                            ].reshape(self.num_of_gtm, self.prod_num)
            if self.aquifer:
                self.aquifer_weights = all_weights[-self.prod_num:]

    def _forward(self, t_0, t_n):
        self.prime_term = (self.lambda_prime*self.prime_rates[t_0:t_n+1]*
                           np.exp(-(self.dt_cum[t_0:t_n+1]-self.dt_cum[t_0])[:, None]/self.tau_prime))
        self.exp_term = ((1. - np.exp(-self.dt[t_0+1:t_n+1, None]/self.tau_weights))*
                         np.exp(-(np.flip(self.dt_cum[t_0+1:t_n+1])-self.dt_cum[t_0+1])[:, None]/
                                self.tau_weights)
                        )
        lambdas_all = ((self.lambda_weights*self.prod_inj_distance_mask))
        self.masked_lambdas = np.ones((self.t_max, self.inj_num, self.prod_num))
#         self.masked_lambdas2 = np.zeros((self.t_max, self.inj_num, self.prod_num))
        if self.gtm_info is not None:
            for ind, (t1, t2, well_ind) in enumerate(self.gtm_info):
                self.masked_lambdas[t1:t2, :, well_ind] = self.betas[ind, well_ind]
                other_inds = np.setdiff1d(np.arange(self.prod_num), [well_ind])
                self.masked_lambdas[t1:t2, :, other_inds] = (1. +
                                                             ((lambdas_all[:, well_ind]*
                                                               (1. - self.betas[ind, well_ind]))[:, None]@
                                                              self.betas[None, ind, other_inds])
                                                            )
        lambdas_all = lambdas_all*self.masked_lambdas# + self.masked_lambdas2[t_0:t_n+1]
        lambdas_shut_sum = np.sum(lambdas_all*self.shut_mask4, axis=2, keepdims=True)
        self.lambdas_vs_time = ((1.-self.shut_mask3)*(lambdas_all*(1 + lambdas_shut_sum)))
        self.conv_inj = np.vstack((np.zeros((1, self.inj_num, self.prod_num)),
                                   [(self.exp_term[-(i+1):, None, :]*
                                     self.injection_rates[t_0+1:t_n+1][:i+1]).sum(0)
                                    for i in range(t_n-t_0)]
                                  ))
        if self.aquifer:
            self.lambdas_vs_time *= (1 - self.aquifer_weights)[None, None, :]
        if self.bhp_term:
            self.conv_pwf = np.vstack((np.zeros((1, self.prod_num)),
                                       (self.tau_weights*self.J_pwf*
                                        [(self.exp_term[-(i+1):]*
                                          (self.dpwf[t_0+1:t_n+1]/
                                           self.dt[t_0+1:t_n+1, None])[:i+1]).sum(0)
                                         for i in range(t_n-t_0)])
                                      ))
        else:
            self.conv_pwf = 0.
        return ((1.-self.shut_mask[t_0:t_n+1])*
                (self.prime_term - self.conv_pwf + (self.lambdas_vs_time[t_0:t_n+1]*self.conv_inj).sum(1))
               )

    def get_prime_term(self, t_0, t_n, tau_prime, lambda_prime):
        self.prime_rates(t_0, t_n)
        return ((1.-self.shut_mask[t_0:t_n+1])*lambda_prime*self.prime_rates*
                np.exp(-(self.dt_cum[t_0:t_n+1]-self.dt_cum[t_0])[:, None]/tau_prime))

    def get_conv_inj(self, t_0, t_n, tau_weights, lambdas):
        exp_term = ((1. - np.exp(-self.dt[t_0+1:t_n+1, None]/tau_weights))*
                    np.exp(-(np.flip(self.dt_cum[t_0+1:t_n+1])-self.dt_cum[t_0+1])[:, None]/
                           tau_weights)
                   )
        conv_inj = np.vstack((np.zeros((1, self.inj_num, self.prod_num)),
                              [(exp_term[-(i+1):, None, :]*
                                self.injection_rates[t_0+1:t_n+1][:i+1]).sum(0)
                               for i in range(t_n-t_0)]
                             ))
        return (1.-self.shut_mask[t_0:t_n+1])*(lambdas[t_0:t_n+1]*conv_inj).sum(1)

    def get_cum_inj(self, t_0, t_n, tau_weights, lambdas):
        cum_inj = (lambdas[t_0:t_n+1]*self.injection_rates[t_0:t_n+1]*
                   self.dt[t_0:t_n+1, None, None]).cumsum(0).sum(1)
        return cum_inj

    def get_conv_pwf(self, t_0, t_n, tau_weights, J_pwf):
        exp_term = ((1. - np.exp(-self.dt[t_0+1:t_n+1, None]/tau_weights))*
                    np.exp(-(np.flip(self.dt_cum[t_0+1:t_n+1])-self.dt_cum[t_0+1])[:, None]/
                           tau_weights)
                   )
        conv_pwf = np.vstack((np.zeros((1, self.prod_num)),
                              (tau_weights*J_pwf*
                               [(exp_term[-(i+1):]*(self.dpwf[t_0+1:t_n+1]/
                                                    self.dt[t_0+1:t_n+1, None])[:i+1]).sum(0)
                                for i in range(t_n-t_0)])
                             ))
        return -(1.-self.shut_mask[t_0:t_n+1])*conv_pwf

    def init_optim_problem(self, t_optim_0, t_optim_n, params=None):
        if params is None:
            self.t_optim_0, self.t_optim_n = t_optim_0, t_optim_n
#             self.tau_prime2 = np.random.uniform(0.1, 1., self.prod_num)
            self.inj_optim_coeffs = np.random.uniform(0., 0.5, (t_optim_n-t_optim_0, self.inj_num))
            self.all_weights2 = self.inj_optim_coeffs.flatten()
        else:
#             self.tau_prime2 = params[:self.prod_num]
            self.inj_optim_coeffs = params.reshape(t_optim_n-t_optim_0, self.inj_num)

    def _forward_optimize(self, t_0, t_n):
        self.prime_term2 = (self.lambda_prime*self.prime_rates[t_0:t_n+1]*
                            np.exp(-(self.dt_cum[t_0:t_n+1]-self.dt_cum[t_0])[:, None]/self.tau_prime))
        self.exp_term2 = ((1. - np.exp(-self.dt[t_0+1:t_n+1, None]/self.tau_weights))*
                          np.exp(-(np.flip(self.dt_cum[t_0+1:t_n+1])-self.dt_cum[t_0+1])[:, None]/
                                 self.tau_weights)
                         )
        self.conv_inj2 = np.vstack((np.zeros((1, self.inj_num, self.prod_num)),
                                   [(self.exp_term2[-(i+1):, None, :]*
                                     (self.inj_max_vals*self.inj_optim_coeffs)[:, :, None][:i+1]).sum(0)
                                    for i in range(t_n-t_0)]
                                  ))
        self.liquid_rates2 = (0. + (self.lambdas_vs_time[[-3]]*self.conv_inj2).sum(1))
        self.cum_injection2 = ((self.lambdas_vs_time[[-3]]*self.inj_max_vals[None, :, None]*
                                self.inj_optim_coeffs[:, :, None]*self.dt[t_0+1:t_n+1, None, None]*
                                self.timestamp).sum(1).cumsum(0))
        self.oil_rates2 = self.liquid_rates2[1:]/(1 + np.exp(-self.a_weights)*self.cum_injection2**self.b_weights)
        return (self.oil_rates2*self.dt[t_0+1:t_n+1, None]).cumsum(0)

    def residuals_optim_problem(self, params):
        self.init_optim_problem(self.t_optim_0, self.t_optim_n, params)
        res = self._forward_optimize(self.t_optim_0, self.t_optim_n)
        return -res.flatten().sum()

    def run_optim_problem(self):
        x0_weights = self.all_weights2.copy()
        bnds = np.array([[0., 1.]]*len(x0_weights))
        result = scipy.optimize.minimize(self.residuals_optim_problem, x0=x0_weights, bounds=bnds, method='SLSQP')

class CRMIP(CRM):
    def __init__(self, t_0, t_n, timestamp, dates, injection_rates, liquid_rates,
                 pwf=None, bhp_term=True, well_distances=None, radius_for_accounting=None,
                 gtm_info=None, lambda_matbal=False, all_weights=None):
        super().__init__(t_0, t_n, timestamp, dates, injection_rates, liquid_rates,
                         pwf, bhp_term, well_distances, radius_for_accounting,
                         gtm_info, lambda_matbal)
        self.set_boundaries()
        self.init_weights(all_weights)

    def set_boundaries(self, bnds_dict={'tau_weights': (0., 20.), 'lambda_weights': (0., 1.),
                                        'J_pwf': (0., 20.), 'betas': (-1., 1.)}):
        self.bnds_dict = bnds_dict
        self.bnds = np.vstack(([self.bnds_dict['lambda_weights']]*self.prod_num,
                               [self.bnds_dict['tau_weights']]*self.prod_num,
                               [self.bnds_dict['lambda_weights']]*self.prod_num*self.inj_num,
                               [self.bnds_dict['tau_weights']]*self.prod_num*self.inj_num))
        if self.bhp_term:
            self.bnds = np.vstack((self.bnds, [self.bnds_dict['J_pwf']]*self.prod_num*self.inj_num))
        if self.gtm_info is not None:
            self.bnds = np.vstack((self.bnds, [self.bnds_dict['betas']]*
                                   self.num_of_gtm*self.inj_num))

    def init_weights(self, all_weights=None):
        if all_weights is None:
            self.lambda_prime = np.random.uniform(0., 1., self.prod_num)
            self.tau_prime = np.random.uniform(0., 1., self.prod_num)
            self.lambda_weights = np.random.uniform(0., 1., (self.inj_num, self.prod_num))
#             self.lambda_weights /= self.lambda_weights.sum(1, keepdims=True)
            self.tau_weights = np.random.uniform(0., 1., (self.inj_num, self.prod_num))
            self.all_weights = np.hstack((self.lambda_prime, self.tau_prime,
                                          self.tau_weights.flatten(), self.lambda_weights.flatten()))
            self.jac_sparsity = np.hstack((np.ones(2*self.prod_num),
                                           self.prod_inj_distance_mask.flatten(),
                                           self.prod_inj_distance_mask.flatten()))
            if self.bhp_term:
                self.J_pwf = np.random.uniform(self.bnds_dict['J_pwf'][0], self.bnds_dict['J_pwf'][1],
                                               (self.inj_num, self.prod_num))
                self.all_weights = np.hstack((self.all_weights, self.J_pwf.flatten()))
                self.jac_sparsity = np.hstack((self.jac_sparsity,
                                               self.prod_inj_distance_mask.flatten()))
            if self.gtm_info is not None:
                self.betas = np.random.uniform(0., 1., (self.num_of_gtm, self.inj_num))
                self.all_weights = np.hstack((self.all_weights, self.betas.flatten()))
                self.jac_sparsity = np.hstack((self.jac_sparsity,
                                               np.ones(self.num_of_gtm*self.inj_num)))
            self.jac_sparsity = self.jac_sparsity[None].repeat(self.prod_num*(self.t_n+1-self.t_0), 0)
        else:
            self.all_weights = all_weights
            self.lambda_prime = all_weights[:self.prod_num]
            self.tau_prime = all_weights[self.prod_num:2*self.prod_num]
            self.lambda_weights = all_weights[2*self.prod_num:self.prod_num*(2+self.inj_num)
                                             ].reshape(self.inj_num, self.prod_num)
            self.tau_weights = all_weights[self.prod_num*(2+self.inj_num):
                                           self.prod_num*(2+2*self.inj_num)
                                          ].reshape(self.inj_num, self.prod_num)
            if self.bhp_term:
                self.J_pwf = all_weights[self.prod_num*(2+2*self.inj_num):
                                         self.prod_num*(2+3*self.inj_num)
                                        ].reshape(self.inj_num, self.prod_num)
                if self.gtm_info is not None:
                    self.betas = all_weights[self.prod_num*(2+3*self.inj_num):
                                             self.prod_num*(2+3*self.inj_num)+
                                             self.num_of_gtm*self.inj_num
                                            ].reshape(self.num_of_gtm, self.inj_num)
            else:
                if self.gtm_info is not None:
                    self.betas = all_weights[self.prod_num*(2+2*self.inj_num):
                                             self.prod_num*(2+2*self.inj_num)+
                                             self.num_of_gtm*self.inj_num
                                            ].reshape(self.num_of_gtm, self.inj_num)

class CRMPWF(CRM):
    def __init__(self, t_0, t_n, timestamp, dates, injection_rates, liquid_rates,
                 pwf=None, bhp_term=True, well_distances=None, radius_for_accounting=None,
                 gtm_info=None, lambda_matbal=False, all_weights=None):
        super().__init__(t_0, t_n, timestamp, dates, injection_rates, liquid_rates,
                         pwf, bhp_term, well_distances, radius_for_accounting,
                         gtm_info, lambda_matbal)
        self.set_boundaries()
        self.init_weights(all_weights)

    def pressure_processing(self, pwf):
        if self.bhp_term:
            self.pwf = pwf
            self.prod_num_pwf = pwf.shape[1]
            self.eye_J_pwf = 2*np.eye(self.prod_num_pwf)[:, :self.prod_num]
            if (self.well_distances[1] is None) or (self.radius_for_accounting[1] is None):
                self.well_distances[1] = np.ones((self.prod_num_pwf, self.prod_num))
                self.radius_for_accounting[1] = 1.
            self.prod_prod_distance_mask = np.sign(pd.DataFrame(self.well_distances[1]).where(
                pd.DataFrame(self.well_distances[1]) <= self.radius_for_accounting[1], 0.).values)
            self.dpwf = np.vstack((np.zeros_like(pwf[0]), pwf[1:] - pwf[:-1]))
            self.dpwf = self.dpwf[:, :, None].repeat(self.prod_num, 2)
            self.dpwf[(self.zero_rows, self.zero_rows_1), self.zero_cols, self.zero_cols] = 0.
            self.dpwf2 = (pwf - pwf[self.t_0])
            self.dpwf2 = self.dpwf2[:, :, None].repeat(self.prod_num, 2)
            self.dpwf2[(self.zero_rows, self.zero_rows_1), self.zero_cols, self.zero_cols] = 0.

#     def pressure_processing(self, pwf):
#         if self.bhp_term:
#             self.pwf = pwf
#             pwf_max_vals = self.pwf[self.t_0+1:].max(0)
#             for el1, el2 in zip(self.zero_rows, self.zero_cols):
#                 self.pwf[el1, el2] = pwf_max_vals[el2]
#             self.prod_num_pwf = pwf.shape[1]
#             self.eye_J_pwf = 2*np.eye(self.prod_num_pwf)[:, :self.prod_num]
#             if self.well_distances[1] is None:
#                 self.well_distances[1] = np.ones((self.prod_num_pwf, self.prod_num))
#                 self.radius_for_accounting[1] = 1.
#             self.prod_prod_distance_mask = (np.sign(pd.DataFrame(self.well_distances[1]).where(
#                 pd.DataFrame(self.well_distances[1]) <= self.radius_for_accounting[1], 0.).values) +
#                                             np.eye(self.prod_num_pwf)[:, :self.prod_num])
#             self.dpwf = np.vstack((self.pwf[0] - pwf_max_vals, pwf[1:] - pwf[:-1]))
#             self.dpwf2 = (pwf - pwf[self.t_0])

    def set_boundaries(self, bnds_dict={'tau_weights': (0., 10.), 'lambda_weights': (0., 1.),
                                        'J_pwf': (0., 10.), 'betas': (0., 10.)}):
        self.bnds_dict = bnds_dict
        self.bnds = np.vstack(([self.bnds_dict['lambda_weights']]*self.prod_num,
                               [self.bnds_dict['tau_weights']]*self.prod_num,
                               [self.bnds_dict['lambda_weights']]*self.prod_num*self.inj_num,
                               [self.bnds_dict['tau_weights']]*self.prod_num))
        if self.bhp_term:
            self.bnds = np.vstack((self.bnds, [self.bnds_dict['J_pwf']]*
                                   self.prod_num_pwf*self.prod_num_pwf))
        if self.gtm_info is not None:
            self.bnds = np.vstack((self.bnds, [self.bnds_dict['betas']]*
                                   self.num_of_gtm*self.prod_num))

    def init_weights(self, all_weights=None):
        if all_weights is None:
            self.lambda_prime = np.random.uniform(0., 1., self.prod_num)
            self.tau_prime = np.random.uniform(0., 1., self.prod_num)
            self.lambda_weights = np.random.uniform(0., 1., (self.inj_num, self.prod_num))
            self.tau_weights = np.random.uniform(0., 1., self.prod_num)
            self.all_weights = np.hstack((self.lambda_prime, self.tau_prime,
                                          self.lambda_weights.flatten(), self.tau_weights))
            self.jac_sparsity = np.hstack((np.ones(2*self.prod_num),
                                           self.prod_inj_distance_mask.flatten(),
                                           np.ones(self.prod_num)))
            if self.bhp_term:
                self.J_pwf = np.random.uniform(self.bnds_dict['J_pwf'][0], self.bnds_dict['J_pwf'][1],
                                               (self.prod_num_pwf, self.prod_num))
                self.all_weights = np.hstack((self.all_weights, self.J_pwf.flatten()))
                self.jac_sparsity = np.hstack((self.jac_sparsity,
                                               self.prod_prod_distance_mask.flatten()))
            if self.gtm_info is not None:
                self.betas = np.random.uniform(0., 1., (self.num_of_gtm, self.prod_num))
                self.all_weights = np.hstack((self.all_weights, self.betas.flatten()))
                self.jac_sparsity = np.hstack((self.jac_sparsity,
                                               np.ones(self.num_of_gtm*self.prod_num)))
            self.jac_sparsity = self.jac_sparsity[None].repeat(self.prod_num*(self.t_n+1-self.t_0), 0)
        else:
            self.all_weights = all_weights
            self.lambda_prime = self.all_weights[:self.prod_num]
            self.tau_prime = self.all_weights[self.prod_num:2*self.prod_num]
            self.lambda_weights = self.all_weights[2*self.prod_num:self.prod_num*(2+self.inj_num)
                                             ].reshape(self.inj_num, self.prod_num)
            self.tau_weights = self.all_weights[self.prod_num*(2+self.inj_num):
                                           self.prod_num*(3+self.inj_num)]
            if self.bhp_term:
                self.J_pwf = self.all_weights[self.prod_num*(3+self.inj_num):
                                         self.prod_num*(3+self.inj_num+self.prod_num_pwf)
                                        ].reshape(self.prod_num_pwf, self.prod_num)
                if self.gtm_info is not None:
                    self.betas = self.all_weights[self.prod_num*(3+self.inj_num+self.prod_num_pwf):
                                             self.prod_num*(3+self.inj_num+self.prod_num_pwf)+
                                             self.num_of_gtm*self.prod_num
                                            ].reshape(self.num_of_gtm, self.prod_num)
            else:
                if self.gtm_info is not None:
                    self.betas = self.all_weights[self.prod_num*(3+self.inj_num):
                                             self.prod_num*(3+self.inj_num)+
                                                            self.num_of_gtm*self.prod_num
                                            ].reshape(self.num_of_gtm, self.prod_num)

#     def _forward(self, t_0, t_n):
#         self.prime_term = (self.liquid_rates[t_0]*self.lambda_prime*
#                            np.exp(-(self.dt_cum[t_0:t_n+1]-self.dt_cum[t_0])[:, None]/self.tau_prime))
#         self.exp_term = ((1. - np.exp(-self.dt[t_0+1:t_n+1, None]/self.tau_weights))*
#                          np.exp(-(np.flip(self.dt_cum[t_0+1:t_n+1])-self.dt_cum[t_0+1])[:, None]/
#                                 self.tau_weights)
#                         )
#         self.conv_inj = np.vstack((np.zeros((1, self.inj_num, self.prod_num)),
#                                    [(self.exp_term[-(i+1):, None, :]*self.injection_rates[t_0+1:t_n+1][:i+1]).sum(0)
#                                     for i in range(t_n-t_0)]
#                                   ))
#         if self.bhp_term:
# #             self.conv_pwf = ((self.J_pwf*self.prod_prod_distance_mask)[None]*self.dpwf2[t_0:t_n+1]).sum(-1)
#             self.conv_pwf = (((2*np.eye(self.prod_num)*self.J_pwf - self.J_pwf)*
#                               self.prod_prod_distance_mask)[None]*
#                              self.dpwf2[t_0:t_n+1]).sum(-1)
#         else:
#             self.conv_pwf = 0.
#         self.masked_lambdas = np.zeros_like(self.injection_rates) + 1.
#         if self.gtm_info is not None:
#             for ind, (t1, t2, well_ind) in enumerate(self.gtm_info):
#                 self.masked_lambdas[t1:t2] = (1 + self.gtm_mask[t1:t2]*self.betas[ind, :, None] +
#                                               (self.gtm_mask[t1:t2] - 1.)*self.betas[ind, :, None]*
#                                               (self.prod_inj_distance_mask*self.lambda_weights)[None, :, well_ind])
#         return (self.prime_term +
#                 (self.lambdas(self.prod_inj_distance_mask*
#                               self.lambda_weights*self.masked_lambdas[t_0:t_n+1])*self.conv_inj).sum(1) -
#                 self.conv_pwf)

    def _forward(self, t_0, t_n):
        self.prime_term = (self.liquid_rates[t_0]*self.lambda_prime*
                           np.exp(-(self.dt_cum[t_0:t_n+1]-self.dt_cum[t_0])[:, None]/self.tau_prime))
        self.exp_term = ((1. - np.exp(-self.dt[t_0+1:t_n+1, None]/self.tau_weights))*
                         np.exp(-(np.flip(self.dt_cum[t_0+1:t_n+1])-self.dt_cum[t_0+1])[:, None]/
                                self.tau_weights)
                        )
        self.masked_lambdas = np.zeros_like(self.injection_rates) + 1.
        if self.gtm_info is not None:
            for ind, (t1, t2, well_ind) in enumerate(self.gtm_info):
                self.masked_lambdas[t1:t2, :, well_ind] = self.betas[ind, well_ind]
                other_inds = np.setdiff1d(np.arange(self.prod_num), [well_ind])
                self.masked_lambdas[t1:t2, :, other_inds] = (1 + self.betas[ind, other_inds]*
                                                             (1 - self.betas[ind, well_ind]))
#         self.masked_lambdas = np.zeros_like(self.injection_rates) + 1.
#         if self.gtm_info is not None:
#             for ind, (t1, t2, well_ind) in enumerate(self.gtm_info):
#                 self.masked_lambdas[t1:t2] = ((1 + self.gtm_mask[t1:t2]*self.betas[ind, :, None] +
#                                               (self.gtm_mask[t1:t2] - 1.)*self.betas[ind, :, None]*
#                                               (self.prod_inj_distance_mask*
#                                                self.lambda_weights)[None, :, well_ind, None])*
#                                               self.prod_inj_distance_mask)
#         self.lambdas_vs_time = self.lambdas((self.lambda_weights*self.masked_lambdas[t_0:t_n+1])*
#                                             (1 + np.sum(self.shut_mask[t_0:t_n+1, None, :]*
#                                                         (self.lambda_weights*self.masked_lambdas[t_0:t_n+1]),
#                                                         axis=2, keepdims=True))
#                                            )
        self.lambdas_vs_time = (self.lambdas(self.lambda_weights)*self.prod_inj_distance_mask*
                                self.masked_lambdas[t_0:t_n+1]*
                                (1 + np.sum(self.shut_mask[t_0:t_n+1, None, :]*
                                            (self.lambdas(self.lambda_weights)*self.prod_inj_distance_mask*
                                             self.masked_lambdas[t_0:t_n+1]), axis=2, keepdims=True))
                               )
        self.conv_inj = np.vstack((np.zeros((1, self.inj_num, self.prod_num)),
                                   [(self.exp_term[-(i+1):, None, :]*
                                     self.injection_rates[t_0+1:t_n+1][:i+1]).sum(0)
                                    for i in range(t_n-t_0)]
                                  ))
        if self.bhp_term:
            self.conv_pwf = np.vstack((np.zeros((1, self.prod_num_pwf, self.prod_num)),
                                       [(self.exp_term[-(i+1):, None, :]*
                                         (self.dpwf[t_0+1:t_n+1]/
                                          self.dt[t_0+1:t_n+1, None, None])[:i+1]).sum(0)
                                        for i in range(t_n-t_0)]
                                      ))
            self.pwf_term = (self.tau_weights*((self.prod_prod_distance_mask*
                                                self.J_pwf*(1 - self.eye_J_pwf))[None]*
                                               self.conv_pwf).sum(1))
        else:
            self.conv_pwf = 0.
        return (self.prime_term + (self.lambdas_vs_time*self.conv_inj).sum(1) + self.pwf_term)

#     def _forward(self, t_0, t_n):
#         self.prime_term = (self.liquid_rates[t_0]*self.lambda_prime*
#                            np.exp(-(self.dt_cum[t_0:t_n+1]-self.dt_cum[t_0])[:, None]/self.tau_prime))
#         self.exp_term = ((1. - np.exp(-self.dt[t_0+1:t_n+1, None]/self.tau_weights))*
#                          np.exp(-(np.flip(self.dt_cum[t_0+1:t_n+1])-self.dt_cum[t_0+1])[:, None]/
#                                 self.tau_weights)
#                         )
#         self.masked_lambdas = np.zeros_like(self.injection_rates) + 1.
#         if self.gtm_info is not None:
#             for ind, (t1, t2, well_ind) in enumerate(self.gtm_info):
#                 self.masked_lambdas[t1:t2] = (1 + self.gtm_mask[t1:t2]*self.betas[ind, :, None] +
#                                               (self.gtm_mask[t1:t2] - 1.)*self.betas[ind, :, None]*
#                                               (self.prod_inj_distance_mask*
#                                                self.lambda_weights)[None, :, well_ind, None])
#         self.masked_inj = (((self.lambda_weights*self.prod_inj_distance_mask*
#                              self.masked_lambdas[t_0+1:t_n+1])*
#                             (1 + np.sum(self.shut_mask[t_0+1:t_n+1, None, :]*
#                                        (self.lambda_weights*self.prod_inj_distance_mask*
#                                         self.masked_lambdas[t_0+1:t_n+1]),
#                                         axis=2, keepdims=True))
#                            )*self.injection_rates[t_0+1:t_n+1])
#         self.conv_inj = np.vstack((np.zeros((1, self.inj_num, self.prod_num)),
#                                    [(self.exp_term[-(i+1):, None, :]*
#                                      self.masked_inj[:i+1]).sum(0)
#                                     for i in range(t_n-t_0)]
#                                   )).sum(1)
#         if self.bhp_term:
#             self.conv_pwf = np.vstack((np.zeros((1, self.prod_num)),
#                                        [(self.exp_term[-(i+1):]*
#                                          (self.dpwf[t_0+1:t_n+1]/self.dt[t_0+1:t_n+1, None])[:i+1]).sum(0)
#                                         for i in range(t_n-t_0)]
#                                       ))
#             self.pwf_term = (self.tau_weights*((self.prod_prod_distance_mask*
#                                                 self.J_pwf*(1 - self.eye_J_pwf))[None]*
#                                                self.conv_pwf[:, :, None]).sum(1))
#         else:
#             self.conv_pwf = 0.
#         return self.prime_term + self.conv_inj + self.pwf_term
