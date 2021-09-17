from IPython import display
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import ipywidgets as widgets
from ipywidgets import interact
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_loss_and_accuracy(loss_history, loss_history_test, semilogplot=True, clear_output=True):
    if clear_output:
        display.clear_output(wait=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(loss_history) > 0:
        if semilogplot:
            ax.semilogy(loss_history, '*-b', label='train')
            ax.semilogy(loss_history_test, '*-r', label='test')
        else:
            ax.plot(loss_history, '*-b', label='train')
            ax.plot(loss_history_test, '*-r', label='test')
        ax.set_title('Loss')
        ax.legend()
        ax.grid()
        ax.set_xlabel('# epochs processed')
        ax.set_ylabel('loss value')
    plt.show()

def show_target_rates(dates, rates, well_names, pwf=None, gtm=None):
    vals = ['ALL'] + list(well_names)
    t_max = rates.shape[0]
    wellname_widget = widgets.Dropdown(options=vals, value=vals[1],
                                       description='Well:', disabled=False)
    t_start_widget = widgets.IntSlider(value=0, min=0, max=t_max-1, step=1,
                                       description='t_start:', disabled=False)
    num_of_wells = len(well_names)
    if num_of_wells == 1: num_of_wells += 1
    def update(wellname, t_start):
        if wellname == 'ALL':
            fig, ax = plt.subplots(num_of_wells, 1, figsize=(18, int(5*num_of_wells)))
            for ind, name in enumerate(well_names):
                max_val = rates[t_start:, ind].max()
                ax[ind].plot(dates[t_start:], rates[t_start:, ind], c='green', label='Liq-'+name)
                ax[ind].set_title('Well ' + name + '. ' + str([ind]))
                if gtm is not None:
                    for x_gtm in np.where(gtm[:, ind])[0]:
                        if x_gtm < t_start:
                            continue
                        else:
                            ax[ind].axvline(x=dates[x_gtm], c='m', ls='--')#, lw=4.)
                            ax[ind].annotate(str(dates[x_gtm].date()) + '\n' + str(x_gtm),
                                             (dates[x_gtm], 0.9*max_val))
                if pwf is not None:
                    ax_new = ax[ind].twinx()
                    ax_new.plot(dates[t_start:], pwf[t_start:, ind], 'k--', label='Pwf-'+name)
                    ax_new.legend(loc='best')
                ax[ind].legend(loc='best')
                ax[ind].grid(True)
        else:
            name, ind = wellname, np.where(well_names == wellname)[0]
            fig, ax = plt.subplots(1, 1, figsize=(18, 5))
            max_val = rates[t_start:, ind].max()
            ax.plot(dates[t_start:], rates[t_start:, ind], c='green', label='Liq-'+name)
            ax.set_title('Well ' + name + '. ' + str(ind))
            if gtm is not None:
                for x_gtm in np.where(gtm[:, ind])[0]:
                    if x_gtm < t_start:
                        continue
                    else:
                        ax.axvline(x=dates[x_gtm], c='m', ls='--')#, lw=4.)
                        ax.annotate(str(dates[x_gtm].date()) + '\n' + str(x_gtm),
                                    (dates[x_gtm], 0.9*max_val))
            if pwf is not None:
                ax_new = ax.twinx()
                ax_new.plot(dates[t_start:], pwf[t_start:, prod_ind[ind]], 'k--', label='Pwf-'+name)
                ax_new.legend(loc='best')
            ax.legend(loc='best')
            ax.grid(True)
    interact(update,
             wellname=wellname_widget,
             t_start=t_start_widget)
    plt.show()

def plot_rates(t_0, t_train, t_max, dates, liq_fact, liq_preds, well_names,
               pwf_fact=None, r_train=None, r_test=None,
               wos=None, figname=None, fluid='Liquid/жидкости', lang='Eng'):
    if lang == 'Eng':
        lab_fact, lab_train, lab_test = 'Target', 'Forecast (train)', 'Forecast (test)'
        lab_dates, lab_liq, lab_press = 'Date', fluid.split('/')[0] + ' rate, m3/day', 'Pressure, bar'
    else:
        lab_fact, lab_train, lab_test = 'Факт', 'Прогноз (обучение)', 'Прогноз (тест)'
        lab_dates, lab_liq, lab_press = 'Дата', 'Дебит ' + fluid.split('/')[-1] + ', м3/день', 'Давление, бар'
    num_of_wells = liq_fact.shape[1]
    if num_of_wells == 1: num_of_wells += 1
    fig, axes = plt.subplots(num_of_wells, 1, figsize=(12, num_of_wells*6))
    for ind in range(liq_fact.shape[1]):
        axes[ind].plot(dates.iloc[t_0:t_max], liq_fact[t_0:t_max, ind], label=lab_fact)
        if r_train is not None:
            axes[ind].plot(dates.iloc[t_0:t_train+1], liq_preds[:t_train+1-t_0, ind],
                           'ro-', markersize=3.5, lw=1., label=lab_train + ' $R^2=%.2f$' % r_train[ind])
        else:
            axes[ind].plot(dates.iloc[t_0:t_train+1], liq_preds[:t_train+1-t_0, ind],
                           'ro-', markersize=3.5, lw=1., label=lab_train)
        if r_test is not None:
            axes[ind].plot(dates.iloc[t_train:t_max], liq_preds[t_train-t_0:t_max-t_0, ind],
                           'go-', markersize=3.5, lw=1., label=lab_test + ' $R^2=%.2f$' % r_test[ind])
        else:
            axes[ind].plot(dates.iloc[t_train:t_max], liq_preds[t_train-t_0:t_max-t_0, ind],
                           'go-', markersize=3.5, lw=1., label=lab_test)
        axes[ind].axvline(x=dates.iloc[t_train], ls='--', lw=2.5, color='orange')
        axes[ind].set_xlabel(lab_dates, fontsize='large')
        axes[ind].set_ylabel(lab_liq, fontsize='large')
        axes[ind].set_title('Well ' + str(well_names[ind]) + '. ' + 'Index ' + str(ind))
        axes[ind].grid()
        axes[ind].legend()
        if pwf_fact is not None:
            ax_inv = axes[ind].twinx()
            ax_inv.plot(dates.iloc[t_0:t_max], pwf_fact[t_0:t_max, ind],
                        ls='--', lw=1.5, color='black', label=lab_press)
            ax_inv.legend()
        if wos is not None:
            max_val = liq_fact[t_0:, ind].max()
            for x_gtm in np.where(wos[:, ind])[0]:
                if x_gtm < t_0:
                    continue
                else:
                    axes[ind].axvline(x=dates[x_gtm], c='m', ls='--')#, lw=4.)
                    axes[ind].annotate(str(dates[x_gtm].date()) + '\n' + str(x_gtm), (dates[x_gtm], 0.9*max_val))
    plt.show()

def plot_lambdas_vs_time(t_0, t_max, dates, lambdas, wnames, lang='Eng'):
    if lang == 'Eng':
        lab_dates, lab_coef = 'Date', 'Coefficient'
    else:
        lab_dates, lab_coef = 'Дата', 'Коэффициент'
    n_inj, n_prod = lambdas.shape[1:]
    num_of_ax = int(n_prod//2)
    if num_of_ax in [0, 1]: num_of_ax += 1
    if n_prod%2 > 0.: num_of_ax += 1
    fig, axes = plt.subplots(num_of_ax, 2, figsize=(16, num_of_ax*5))
    for ind in range(n_prod):
        ind_right, ind_left = ind//2, ind%2
        for el in np.arange(1, n_inj+1):
            axes[ind_right, ind_left].plot(dates.iloc[t_0:t_max],
                                           lambdas[:t_max-t_0, el-1, ind],
                                           label='$\lambda_{%.0f%.0f}(t)$'%(el, ind+1))
        axes[ind_right, ind_left].grid()
        axes[ind_right, ind_left].legend(fontsize='large', loc='best')
        axes[ind_right, ind_left].set_xlabel(lab_dates, fontsize='x-large')
        axes[ind_right, ind_left].set_ylabel(lab_coef + ' $\lambda_{%.0fj}$'%(ind+1), fontsize='x-large')
        axes[ind_right, ind_left].set_title('Well ' + wnames[ind], fontsize='x-large')
#     fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

def plot_weight_map(weights0, prod_wells, inj_wells, prod_wells_coords, inj_wells_coords,
                            prod_inj_distances, radius, bound_vals=(0., 1.), r2_vals=None):
    n_timesteps = weights0.shape[0]
    timestep_widget = widgets.IntSlider(min=0, max=n_timesteps-1, step=1, value=0)

    @interact(
        timestep=timestep_widget
    )

    def update(timestep):
        weights = weights0[timestep]
        cmap = plt.cm.Greens
        cNorm  = colors.Normalize(vmin=0., vmax=np.min([1., bound_vals[1]]))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        for wname, (ind, point) in zip(prod_wells, enumerate(prod_wells_coords)):
            try:
                text_el = wname.split('-')[1]
            except IndexError:
                text_el = wname
            if r2_vals is not None:
                text_el += '-'+str(r2_vals[ind].round(1))
            ax.plot([point[0]], [point[1]], 'r^', markersize=11., mec='k')
            ax.text(point[0], point[1], text_el, fontsize='x-large')
            mask = prod_inj_distances <= radius
            colorVal = scalarMap.to_rgba((weights*mask)[:, ind])
            for ind2, (dx, dy) in enumerate((inj_wells_coords - point)*mask[:, [ind]]):
                if (weights*mask)[ind2, ind] > bound_vals[0]:
                    ax.arrow(point[0], point[1], dx, dy, lw=7., animated=True, color=colorVal[ind2])
            ax.plot(inj_wells_coords[:, 0], inj_wells_coords[:, 1], 'b*', markersize=13., mec='k')
            for kk, wname, point in zip(np.arange(len(inj_wells)), inj_wells, inj_wells_coords):
                try:
                    text_el = wname.split('-')[1]
                except IndexError:
                    text_el = wname
                ax.text(point[0], point[1], text_el, fontsize='x-large')
        ax.set_xlabel('X coordinate', fontsize='x-large')
        ax.set_ylabel('Y coordinate', fontsize='x-large')
        ax.set_title('Lambda weights map ', fontsize='x-large')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb1 = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=cNorm, orientation='vertical')
        plt.show()

def plot_weight_map_per_layer(weights0, all_layers, prod_wells, inj_wells, prod_wells_coords, inj_wells_coords,
                            prod_inj_distances, radius, bound_vals=(0., 1.), r2_vals=None):
    n_timesteps = weights0.shape[0]

    timestep_widget = widgets.IntSlider(min=0, max=n_timesteps-1, step=1, value=0)
    layer_widget = widgets.Dropdown(options=all_layers, index=0, description='Layer:', disabled=False)

    prod_inds_per_layer = {}
    for layer in all_layers:
        prod_inds_per_layer[layer] = [i for i, name in enumerate(prod_wells) if layer in name]

    def update(timestep, layer):
        weights = weights0[timestep][:, prod_inds_per_layer[layer]]
        cmap = plt.cm.Greens
        cNorm  = colors.Normalize(vmin=0., vmax=np.min([1., bound_vals[1]]))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        for wname, (ind, point) in zip(prod_wells[prod_inds_per_layer[layer]],
                                       enumerate(prod_wells_coords[prod_inds_per_layer[layer]])):
            try:
                text_el = wname.split('-')[1]
            except IndexError:
                text_el = wname
            if r2_vals is not None:
                text_el += '-'+str(r2_vals[ind].round(1))
            ax.plot([point[0]], [point[1]], 'r^', markersize=11., mec='k')
            ax.text(point[0], point[1], text_el, fontsize='x-large')
            mask = prod_inj_distances[:, prod_inds_per_layer[layer]] <= radius
            colorVal = scalarMap.to_rgba((weights*mask)[:, ind])
            for ind2, (dx, dy) in enumerate((inj_wells_coords - point)*mask[:, [ind]]):
                if (weights*mask)[ind2, ind] > bound_vals[0]:
                    ax.arrow(point[0], point[1], dx, dy, lw=7., animated=True, color=colorVal[ind2])
            ax.plot(inj_wells_coords[:, 0], inj_wells_coords[:, 1], 'b*', markersize=13., mec='k')
            for kk, wname, point in zip(np.arange(len(inj_wells)), inj_wells, inj_wells_coords):
                try:
                    text_el = wname.split('-')[1]
                except IndexError:
                    text_el = wname
                ax.text(point[0], point[1], text_el, fontsize='x-large')
        ax.set_xlabel('X coordinate', fontsize='x-large')
        ax.set_ylabel('Y coordinate', fontsize='x-large')
        ax.set_title('Lambda weights map. Layer ' + layer, fontsize='x-large')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb1 = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=cNorm, orientation='vertical')
    interact(
        update,
        timestep=timestep_widget,
        layer=layer_widget
    )
    plt.show()

def show_layers(all_layers, prod_wells_coords, inj_wells_coords):
    layer_widget = widgets.Dropdown(options=all_layers, index=0, description='Layer:', disabled=False)
    def update(layer):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for wind, name, xx, yy in zip(np.arange(len(prod_wells)), prod_wells, prod_wells_coords[:, 0], prod_wells_coords[:, 1]):
            wname, lname  = name.split('-')[1:3]
            if lname in layer:
                ax.plot(prod_wells_coords[[wind], 0], prod_wells_coords[[wind], 1], 'r^', markersize=7., mec='k')
                ax.annotate(wname, (xx, yy))
        for wind, name, xx, yy in zip(np.arange(len(inj_wells)), inj_wells, inj_wells_coords[:, 0], inj_wells_coords[:, 1]):
            wname, lname  = name.split('-')[1:3]
            if lname in layer:
                ax.plot(inj_wells_coords[[wind], 0], inj_wells_coords[[wind], 1], 'g*', markersize=9., mec='k')
                ax.annotate(wname, (xx, yy))
        ax.set_title('Layer ' + layer, fontsize='large')
        (coordx_min, coordy_min), (coordx_max, coordy_max) = prod_wells_coords.min(axis=0), prod_wells_coords.max(axis=0)
        ax.set_xlim((coordx_min, coordx_max))
        ax.set_ylim((coordy_min, coordy_max))
        ax.set_xlabel('X Coordinate', fontsize='large')
        ax.set_ylabel('Y Coordinate', fontsize='large')
        ax.grid()
    interact(update, layer=layer_widget)
    plt.show()

def jarvismarch(A):
    def rotate(A,B,C):
        return (B[0]-A[0])*(C[1]-B[1])-(B[1]-A[1])*(C[0]-B[0])
    n = len(A)
    P = list(np.arange(n))
    # start point
    for i in range(1, n):
        if A[P[i]][0]<A[P[0]][0]:
            P[i], P[0] = P[0], P[i]  
    H = [P[0]]
    del P[0]
    P.append(H[0])
    while True:
        right = 0
        for i in range(1, len(P)):
            if rotate(A[H[-1]],A[P[right]],A[P[i]])<0:
                right = i
        if P[right]==H[0]: 
            break
        else:
            H.append(P[right])
            del P[right]
    return H

def plot_sweep_efficiency(weights, prod_wells, inj_wells, prod_wells_coords, inj_wells_coords):
    t_max = weights.shape[0]
    t_step_widget = widgets.IntSlider(value=t_max-1, min=0, max=t_max-1, step=1,
                                      description='t:', disabled=False)
    weights[weights > 1.] = 1.
    weights_norm = weights.copy()
    def update(t_step):
        sweep_zone = [inj_wells_coords]
        for prod_ind, prod_xy in enumerate(prod_wells_coords):
            if len(prod_wells_coords) == 1:
                sweep_zone.append(inj_wells_coords + weights_norm[t_step, :, None, prod_ind]*
                                  (prod_xy - inj_wells_coords)*np.array([[1., 0.]]*1))
                sweep_zone.append(inj_wells_coords + weights_norm[t_step, :, None, prod_ind]*
                                  (prod_xy - inj_wells_coords))
                sweep_zone.append(inj_wells_coords + weights_norm[t_step, :, None, prod_ind]*
                                  (prod_xy - inj_wells_coords)*np.array([[0., 1.]]*1))
            else:
                sweep_zone.append(inj_wells_coords +
                                  weights_norm[t_step, :, None, prod_ind]*(prod_xy - inj_wells_coords))
        sweep_zone = np.transpose(np.array(sweep_zone), axes=(1, 0, 2))

        fig = plt.figure(figsize=(12, 10))
        for elems in sweep_zone:
#             plt.plot(elems[:, 0], elems[:, 1], 'bo')
            inds = jarvismarch(elems)
            xx, yy = elems[inds].T
            plt.fill(xx, yy, 'b', alpha=0.35)
        plt.plot(prod_wells_coords[:, 0], prod_wells_coords[:, 1], 'r^', markersize=14., mec='k')
        plt.plot(inj_wells_coords[:, 0], inj_wells_coords[:, 1], 'b*', markersize=17., mec='k')
        for wname, point in zip(inj_wells, inj_wells_coords):
            try:
                text_el = wname.split('-')[1]
            except IndexError:
                text_el = wname
            plt.text(point[0], point[1], text_el, fontsize='large')
        for wname, point in zip(prod_wells, prod_wells_coords):
            try:
                text_el = wname.split('-')[1]
            except IndexError:
                text_el = wname
            plt.text(point[0], point[1], text_el, fontsize='large')
    interact(update, t_step=t_step_widget)
    plt.show()

def plot_sweep_efficiency_per_layer(weights, all_layers, prod_wells, inj_wells, prod_wells_coords, inj_wells_coords):
    layer_widget = widgets.Dropdown(options=all_layers, index=0, description='Layer:', disabled=False)

    weights[weights > 1.] = 1.
    weights_norm = weights.copy() #/weights.max((1, 2), keepdims=True)
    t_max = weights.shape[0]
    t_step_widget = widgets.IntSlider(value=t_max-1, min=0, max=t_max-1, step=1,
                                      description='t:', disabled=False)
    prod_inds_per_layer = {}
#     inj_inds_per_layer = {}
    for layer in all_layers:
        prod_inds_per_layer[layer] = [i for i, name in enumerate(prod_wells) if layer in name]
#         inj_inds_per_layer[layer] = [i for i, name in enumerate(inj_wells) if layer in name]
    sweep_zone_per_layer = {}
    def update(t_step, layer):
        sweep_zone = [inj_wells_coords]
        for prod_ind, prod_xy in zip(prod_inds_per_layer[layer], prod_wells_coords[prod_inds_per_layer[layer]]):
            if len(prod_wells_coords) == 1:
                sweep_zone.append(inj_wells_coords + weights_norm[t_step, :, None, prod_ind]*
                                  (prod_xy - inj_wells_coords)*np.array([[1., 0.]]*1))
                sweep_zone.append(inj_wells_coords + weights_norm[t_step, :, None, prod_ind]*
                                  (prod_xy - inj_wells_coords))
                sweep_zone.append(inj_wells_coords + weights_norm[t_step, :, None, prod_ind]*
                                  (prod_xy - inj_wells_coords)*np.array([[0., 1.]]*1))
            else:
                sweep_zone.append(inj_wells_coords + weights_norm[t_step, :, None, prod_ind]*
                                  (prod_xy - inj_wells_coords))
        sweep_zone_per_layer[layer] = np.transpose(np.array(sweep_zone), axes=(1, 0, 2))
        fig = plt.figure(figsize=(12, 10))
        for elems in sweep_zone_per_layer[layer]:
            plt.plot(elems[:, 0], elems[:, 1], 'bo', markersize=3.5)
            inds = jarvismarch(elems)
            xx, yy = elems[inds].T
            plt.fill(xx, yy, 'b', alpha=0.35)
        plt.plot(prod_wells_coords[prod_inds_per_layer[layer], 0],
                 prod_wells_coords[prod_inds_per_layer[layer], 1], 'r^', markersize=14., mec='k')
        plt.plot(inj_wells_coords[:, 0], inj_wells_coords[:, 1], 'b*', markersize=17., mec='k')
        for wname, point in zip(inj_wells, inj_wells_coords):
            try:
                text_el = wname.split('-')[1]
            except IndexError:
                text_el = wname
            plt.text(point[0], point[1], text_el, fontsize='large')
        for wname, point in zip(prod_wells[prod_inds_per_layer[layer]],
                                prod_wells_coords[prod_inds_per_layer[layer]]):
            try:
                text_el = wname.split('-')[1]
            except IndexError:
                text_el = wname
            plt.text(point[0], point[1], text_el, fontsize='large')
        plt.xlabel('X Coordinate', fontsize='large')
        plt.ylabel('Y Coordinate', fontsize='large')
        plt.title('Layer ' + layer, fontsize='large')
    interact(update, t_step=t_step_widget, layer=layer_widget)
    plt.show()
