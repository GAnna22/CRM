from IPython import display
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import ipywidgets as widgets
from ipywidgets import interact
from mpl_toolkits.axes_grid1 import make_axes_locatable

def choose_xy_ranges(filepath):
    ws_coords, ws_bhp, ws_inj, ws_prd, workovers = (pd.read_csv(filepath+'/ws_coords.csv'),
                                                    pd.read_csv(filepath+'/ws_bhp.csv'),
                                                    pd.read_csv(filepath+'/ws_inj.csv'),
                                                    pd.read_csv(filepath+'/ws_prd.csv'),
                                                    pd.read_csv(filepath+'/workovers.csv')
                                                   )
    ws_coords2 = pd.read_excel(filepath+'/Complete Well Data Report.xlsx')
    x_all, y_all = ws_coords.loc[:, ['XCoord', 'YCoord']].values.T
    x_all2, y_all2 = ws_coords2.loc[:,  ['X Coordinates of Extraction Point(UTM42)',
                                         'Y Coordinates of Extraction Point(UTM42)']].values.T
    x_min, x_max = np.hstack((x_all, x_all2)), np.hstack((x_all, x_all2))
    x_min, x_max = x_min[~np.isnan(x_min)].min(), x_max[~np.isnan(x_max)].max()
    y_min, y_max = np.hstack((y_all, y_all2)), np.hstack((y_all, y_all2))
    y_min, y_max = y_min[~np.isnan(y_min)].min(), y_max[~np.isnan(y_max)].max()

    x_bounds_widget = widgets.SelectionRangeSlider(options=np.arange(x_min, x_max, 30).astype(int),
                                                   index=(0, 29), description='x range', disabled=False)
    y_bounds_widget = widgets.SelectionRangeSlider(options=np.linspace(y_min, y_max, 30).astype(int),
                                                   index=(0, 29), description='y range', disabled=False)
    def update(x_bounds, y_bounds, accept):
        xx, yy, wellnames = ws_coords2.loc[
            ((ws_coords2['X Coordinates of Extraction Point(UTM42)'] <= x_bounds[1])*1 +
             (ws_coords2['X Coordinates of Extraction Point(UTM42)'] >= x_bounds[0])*1 +
             (ws_coords2['Y Coordinates of Extraction Point(UTM42)'] <= y_bounds[1])*1 +
             (ws_coords2['Y Coordinates of Extraction Point(UTM42)'] >= y_bounds[0])*1) == 4,
            ['X Coordinates of Extraction Point(UTM42)',
             'Y Coordinates of Extraction Point(UTM42)', 'Well Name']].values.T

        xx2, yy2, wellnames2 = ws_coords.loc[((ws_coords['XCoord'] <= x_bounds[1])*1 +
                                              (ws_coords['XCoord'] >= x_bounds[0])*1 +
                                              (ws_coords['YCoord'] <= y_bounds[1])*1 +
                                              (ws_coords['YCoord'] >= y_bounds[0])*1) == 4,
                                             ['XCoord', 'YCoord', 'Conduit']].values.T
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(x_all2, y_all2, 'bo', markersize=5., mec='k', alpha=0.5)
        ax.scatter(xx, yy, color='orange')
        ax.scatter(xx2, yy2, color='orange')
        ax.axvline(x=x_bounds[0], ls='--', color='black')
        ax.axvline(x=x_bounds[1], ls='--', color='black')
        ax.axhline(y=y_bounds[0], ls='--', color='black')
        ax.axhline(y=y_bounds[1], ls='--', color='black')
        ax.grid()
        if accept:
            return {'x': x_bounds, 'y': y_bounds}
    interact(update,
             x_bounds=x_bounds_widget,
             y_bounds=y_bounds_widget,
             accept=False
    )
    plt.show()

def preprocessing_spd(filepath, coord_bounds):
    ws_coords, ws_bhp, ws_inj, ws_prd, workovers = (pd.read_csv(filepath+'/ws_coords.csv'),
                                                    pd.read_csv(filepath+'/ws_bhp.csv'),
                                                    pd.read_csv(filepath+'/ws_inj.csv'),
                                                    pd.read_csv(filepath+'/ws_prd.csv'),
                                                    pd.read_csv(filepath+'/workovers.csv')
                                                   )
    ws_coords2 = pd.read_excel(filepath+'/Complete Well Data Report.xlsx')
#     workovers = pd.read_csv(filepath+'/workovers.csv')
#     workovers['Date'] = pd.to_datetime(workovers['Date'])
#     for i in range(workovers['Date'].shape[0]):
#         tt = workovers['Date'].iloc[i]
#         dd, mm, yy = tt.day, tt.month, tt.year
#         if dd > 15:
#             if mm == 12:
#                 mm = 1
#                 yy += 1
#             else:
#                 mm += 1
#             workovers['Date'].iloc[i] = pd.to_datetime('/'.join([str(mm), '1', str(yy)]))
#         else:
#             if mm == 1:
#                 mm = 12
#                 yy -= 1
#             workovers['Date'].iloc[i] = pd.to_datetime('/'.join([str(mm), '1', str(yy)]))
#     workovers['WO'] = 1.
#     x_min, x_max = np.hstack((x_all, x_all2)), np.hstack((x_all, x_all2))
#     x_min, x_max = int(x_min[~np.isnan(x_min)].min()), int(x_max[~np.isnan(x_max)].max())
#     y_min, y_max = np.hstack((y_all, y_all2)), np.hstack((y_all, y_all2))
#     y_min, y_max = int(y_min[~np.isnan(y_min)].min()), int(y_max[~np.isnan(y_max)].max())

#     x_bounds_widget = widgets.SelectionRangeSlider(options=np.arange(x_min, x_max, 20),
#                                                    index=(0, 19), description='x bounds', disabled=False)
#     y_bounds_widget = widgets.SelectionRangeSlider(options=np.linspace(y_min, y_max, 20),
#                                                    index=(0, 19), description='y bounds', disabled=False)
#     @interact(x_bounds=x_bounds_widget,
#               y_bounds=y_bounds_widget,
#               accept=False
#     )
    xx, yy, wellnames = ws_coords2.loc[
        ((ws_coords2['X Coordinates of Extraction Point(UTM42)'] <= coord_bounds['x'][1])*1 +
         (ws_coords2['X Coordinates of Extraction Point(UTM42)'] >= coord_bounds['x'][0])*1 +
         (ws_coords2['Y Coordinates of Extraction Point(UTM42)'] <= coord_bounds['y'][1])*1 +
         (ws_coords2['Y Coordinates of Extraction Point(UTM42)'] >= coord_bounds['y'][0])*1) == 4,
        ['X Coordinates of Extraction Point(UTM42)',
         'Y Coordinates of Extraction Point(UTM42)', 'Well Name']].values.T

    xx2, yy2, wellnames2 = ws_coords.loc[((ws_coords['XCoord'] <= coord_bounds['x'][1])*1 +
                                          (ws_coords['XCoord'] >= coord_bounds['x'][0])*1 +
                                          (ws_coords['YCoord'] <= coord_bounds['y'][1])*1 +
                                          (ws_coords['YCoord'] >= coord_bounds['y'][0])*1) == 4,
                                         ['XCoord', 'YCoord', 'Conduit']].values.T
    wells_to_consider0 = np.unique(np.hstack((wellnames, wellnames2)))
    ws_bhp['Date'] = pd.to_datetime(ws_bhp['Date'])
#     ws_inj['Date'], ws_prd['Date'] = pd.to_datetime(ws_inj['Date']), pd.to_datetime(ws_prd['Date'])
#     ws_inj['Conduit'] = list(map(lambda elem: elem.split('-AS')[0], ws_inj['UniqueId'].values))
#     ws_prd['Conduit'] = list(map(lambda elem: elem.split('-AS')[0], ws_prd['UniqueId'].values))
#     ws_prd_inj = pd.concat((ws_prd, ws_inj)).reset_index(drop=True)
#     ws_prd_inj.loc[ws_prd_inj['Oil'] > 0., 'Volume'] = 0.
#     ws_prd_inj.loc[ws_prd_inj['Volume'] > 0., ['Oil', 'Gas', 'Water']] = np.zeros(3)
#     field_data = pd.merge(ws_prd_inj, ws_coords[['UniqueId', 'Reservoir', 'MEGA_BLOCK']],
#                           how='left', on='UniqueId')
#     field_data[['Reservoir', 'MEGA_BLOCK']] = field_data[['Reservoir', 'MEGA_BLOCK']].fillna(0)
#     field_data = pd.merge(field_data, ws_bhp, how='left', on=['Conduit', 'Date']).reset_index(drop=True)
#     field_data.loc[field_data['Volume'] > 0., 'BhpEchometerTopPerfDaily'] = 0.
#     field_data['Liquid'] = field_data['Oil'] + field_data['Water']
#         field_data = pd.merge(field_data, workovers[['Conduit', 'Date', 'WO', 'WO_DESCR']],
#                               how='left', on=['Conduit', 'Date']).reset_index(drop=True)
#         field_data.to_csv('dataset_spd_ws/field_data.csv')
    field_data = pd.read_csv(filepath+'/field_data.csv', index_col=0)
    field_data['Date'] = pd.to_datetime(field_data['Date'])
    wells_to_consider = np.unique(np.array([elem1 for elem1, elem2 in
                                            zip(field_data['UniqueId'], field_data['Conduit'])
                                            if elem2 in wells_to_consider0]))
    wells_to_exclude = np.intersect1d(np.hstack((ws_coords.loc[ws_coords['WellType']=='OBSERVATION', 'UniqueId'],
                                                 ws_coords.loc[ws_coords['WellType']=='EXPLORATION', 'UniqueId'])),
                                      field_data['UniqueId'].unique())
    wells_to_consider = np.setdiff1d(wells_to_consider, wells_to_exclude)

    final_df = pd.merge(field_data.loc[field_data['UniqueId']==wells_to_consider[0],
                                       ['Date', 'Days', 'Oil', 'Water', 'Gas',
                                        'Liquid', 'Volume', 'BhpEchometerTopPerfDaily', 'WO']],
                        field_data.loc[field_data['UniqueId']==wells_to_consider[1],
                                       ['Date', 'Days', 'Oil', 'Water', 'Gas',
                                        'Liquid', 'Volume', 'BhpEchometerTopPerfDaily', 'WO']],
                        how='outer', on='Date').reset_index(drop=True)
    for well_name in wells_to_consider[2:]:
        final_df = pd.merge(final_df, field_data.loc[field_data['UniqueId']==well_name,
                                                     ['Date', 'Days', 'Oil', 'Water', 'Gas', 'Liquid',
                                                      'Volume', 'BhpEchometerTopPerfDaily', 'WO']],
                            how='outer', on='Date').reset_index(drop=True)
    final_df = final_df.sort_values(by='Date').reset_index(drop=True)
    for (ind, well_name) in zip(np.arange(len(final_df.columns))[1::8], wells_to_consider):
        new_cols = [item + well_name for item in
                    ['Days-', 'Oil-', 'Wat-', 'Gas-', 'Liq-', 'Inj-', 'Pwf-', 'WO-']
                   ]
        for k, col in enumerate(new_cols):
            if 'WO' in col:
                final_df[col] = final_df.iloc[:, ind+k].fillna(0)
                continue
            final_df[col] = final_df.iloc[:, ind+k]
    final_df.drop(final_df.columns[
        np.arange(len(final_df.columns))[1:1+8*len(wells_to_consider)]], axis=1, inplace=True)
    final_df = final_df.groupby(by='Date', as_index=False).mean()

#         gtm_type_df = pd.merge(field_data.loc[field_data['UniqueId']==wells_to_consider[0], ['Date', 'WO_TYPE']],
#                                field_data.loc[field_data['UniqueId']==wells_to_consider[1], ['Date', 'WO_TYPE']],
#                                how='outer', on='Date').reset_index(drop=True)
#         for well_name in wells_to_consider[2:]:
#             gtm_type_df = pd.merge(gtm_type_df, field_data.loc[field_data['UniqueId']==well_name, ['Date', 'WO_TYPE']],
#                                    how='outer', on='Date').reset_index(drop=True)
#         gtm_type_df = gtm_type_df.sort_values(by='Date').reset_index(drop=True)
#         for (ind, well_name) in zip(np.arange(len(gtm_type_df.columns))[1:], wells_to_consider):
#             new_col = 'WO-' + well_name
#             gtm_type_df[new_col] = gtm_type_df.iloc[:, ind].fillna(0)
#         gtm_type_df.drop(gtm_type_df.columns[
#             np.arange(len(gtm_type_df.columns))[1:1+len(wells_to_consider)]], axis=1, inplace=True)
#         gtm_type_df = gtm_type_df.groupby(by='Date', as_index=False).min()
    return final_df, wells_to_consider#, gtm_type_df

def choose_t_start(final_df):
    liq_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Liq']]
    inj_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Inj']]
    pwf_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Pwf']]
    def update(t_start):
        fig, ax_hist = plt.subplots(1, 3, figsize=(16, 16))
        ax_hist[0].imshow(np.sign(liq_df.loc[:].values.T))
        ax_hist[1].imshow(np.sign(inj_df.loc[:].values.T))
        ax_hist[2].imshow(np.sign(pwf_df.loc[:].values.T))
        ax_hist[0].axvline(x=t_start, lw=2.5, ls='--', color='red')
        ax_hist[0].set_title('Liquid production')
        ax_hist[0].set_xlabel('timestep')
        ax_hist[0].set_ylabel('Well №')
        ax_hist[1].axvline(x=t_start, lw=2.5, ls='--', color='red')
        ax_hist[1].set_title('Injection')
        ax_hist[1].set_xlabel('timestep')
        ax_hist[2].axvline(x=t_start, lw=2.5, ls='--', color='red')
        ax_hist[2].set_title('BHPressure')
        ax_hist[2].set_xlabel('timestep')
        fig.tight_layout()
    t_max = final_df.shape[0]
    t_start_widget = widgets.IntSlider(value=0, min=0, max=t_max-1, step=1,
                                       description='t_start:', disabled=False)
    interact(update, t_start=t_start_widget)
    plt.show()

def preprocessing_spd_final(t_start, final_df, wells_to_consider,
                            zerowing=True, cut_coeff=4/5, use_days=True, gtm_type_df=None):
    final_df = final_df.loc[t_start:].reset_index(drop=True)
#         gtm_type_df = gtm_type_df.iloc[t_start:, 1:]
    for well_name in wells_to_consider:
        new_cols = np.array([item + well_name for item in
                             ['Days-', 'Oil-', 'Wat-', 'Gas-', 'Liq-', 'Inj-', 'Pwf-']])
        notna_inds = np.where(final_df.loc[:, new_cols[4]].notna())[0]
        if len(notna_inds) != 0:
            isna_inds = np.where(final_df.loc[:notna_inds[0], new_cols[4]].isna())[0]
            if len(isna_inds) != 0:
                final_df.loc[:notna_inds[0], new_cols] = 0.
            try:
                final_df.loc[notna_inds[-1]+1:, new_cols] = 0.
            except:
                pass
            pwf_isna_inds = np.where(final_df.loc[notna_inds[0]:, new_cols[-1]].isna())[0]
            if len(pwf_isna_inds) != 0:
                pwf_isna_inds += notna_inds[0]
                pwf_notna_inds = np.where(final_df.loc[pwf_isna_inds[0]:, new_cols[-1]].notna())[0]
                if len(pwf_notna_inds) != 0:
                    pwf_notna_inds += pwf_isna_inds[0]
                    if zerowing:
                        final_df.loc[pwf_isna_inds[0]:pwf_notna_inds[0], new_cols[[0, 6]]] = (final_df.loc[
                        pwf_isna_inds[0]:pwf_notna_inds[0], new_cols[[0, 6]]].fillna(method='bfill')
                                                                                            )
                    else:
                        final_df.loc[pwf_isna_inds[0]:pwf_notna_inds[0], new_cols[[0, 5, 6]]] = (final_df.loc[
                        pwf_isna_inds[0]:pwf_notna_inds[0], new_cols[[0, 5, 6]]].fillna(method='bfill')
                                                                                     )
    if zerowing:
        final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Pwf']
                           ] = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Pwf']
                           ].fillna(method='ffill').fillna(0.)
        final_df = final_df.fillna(0.)
    else:
        final_df = final_df.fillna(method='ffill').fillna(0.)
    # cut_coeff - кол-во данных (1 - cut_coeff) от всего интервала
    oil_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Oil']]
    wat_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Wat']]
    gas_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Gas']]
    liq_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Liq']]
    inj_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Inj']]
    pwf_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Pwf']]
    gtm_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:2] == 'WO']]
    days_df = final_df.loc[:, [elem for elem in final_df.columns if elem[:3] == 'Day']]
    # days_vals = np.hstack(([31.], pd.to_timedelta(dates.values[1:] - dates.values[:-1]).days))[:, None]
    t_all = final_df.shape[0]
    prod_inds = np.arange(liq_df.shape[1])[(liq_df == 0.).sum() <= cut_coeff*t_all]
    inj_inds = np.arange(inj_df.shape[1])[(inj_df == 0.).sum() <= cut_coeff*t_all]
    prod_wells = np.array(['-'.join(name.split('-')[1:]) for name in liq_df.columns[prod_inds].values])
    inj_wells = np.array(['-'.join(name.split('-')[1:]) for name in inj_df.columns[inj_inds].values])
    fig, ax_hist = plt.subplots(2, 3, figsize=(16, 2*6))
    ax_hist[0,0].imshow(np.sign(oil_df.iloc[:, prod_inds].values.T))
    ax_hist[0,1].imshow(np.sign(wat_df.iloc[:, prod_inds].values.T))
    ax_hist[0,2].imshow(np.sign(gas_df.iloc[:, prod_inds].values.T))
    ax_hist[1,0].imshow(np.sign(liq_df.iloc[:, prod_inds].values.T))
    ax_hist[1,1].imshow(np.sign(inj_df.iloc[:, inj_inds].values.T))
    ax_hist[1,2].imshow(np.sign(pwf_df.iloc[:, prod_inds].values.T))
    ax_hist[0,0].set_title('Oil rate')
    ax_hist[0,1].set_title('Water rate')
    ax_hist[0,2].set_title('Gas rate')
    ax_hist[1,0].set_title('Liquid rate')
    ax_hist[1,1].set_title('Injection rate')
    ax_hist[1,2].set_title('BHPressure')
    ax_hist[1,0].set_xlabel('timestep')
    ax_hist[1,1].set_xlabel('timestep')
    ax_hist[1,2].set_xlabel('timestep')
    ax_hist[0,0].set_ylabel('Well №')
    ax_hist[1,0].set_ylabel('Well №')
    fig.tight_layout()
    plt.show()
    dates = final_df.loc[:, 'Date']
    if use_days:
        days_prod = days_df.iloc[:, prod_inds].replace(0., 1.).values
        days_inj = days_df.iloc[:, inj_inds].replace(0., 1.).values
    else:
        days_prod = np.hstack(([30.], pd.to_timedelta(dates.values[1:] - dates.values[:-1]).days))[:, None]
        days_inj = np.hstack(([30.], pd.to_timedelta(dates.values[1:] - dates.values[:-1]).days))[:, None]
    oil_output = oil_df.iloc[:, prod_inds].values/days_prod
    wat_output = wat_df.iloc[:, prod_inds].values/days_prod
    gas_output = gas_df.iloc[:, prod_inds].values/days_prod
    liq_output = oil_output + wat_output # liq_df.iloc[:, prod_inds].values/days_prod
    gtm_prod = gtm_df.iloc[:, prod_inds].values
    inj_input = inj_df.iloc[:, inj_inds].values/days_inj
#     gtm_inj = gtm_df.iloc[:, inj_inds].values
    pwf = pwf_df.iloc[:, prod_inds].values
    return (prod_wells, inj_wells), (dates, oil_output, wat_output, gas_output, liq_output, gtm_prod, inj_input, pwf)

def well_coordinates(filepath, prod_wells, inj_wells):
    ws_coords = pd.read_csv(filepath+'/ws_coords.csv')
    ws_coords2 = pd.read_excel(filepath+'/Complete Well Data Report.xlsx')
    x_all2, y_all2 = ws_coords2.loc[:,  ['X Coordinates of Extraction Point(UTM42)',
                                         'Y Coordinates of Extraction Point(UTM42)']].values.T
    prod_wells_coords = []
    for ind, elem in enumerate(prod_wells):
        try:
            prod_wells_coords.append(ws_coords2.loc[ws_coords2['Well Name'] == '-'.join(elem.split('-')[:2]),
                                                    ['X Coordinates of Extraction Point(UTM42)',
                                                     'Y Coordinates of Extraction Point(UTM42)']].values[0])
        except:
            try:
                prod_wells_coords.append(ws_coords.loc[ws_coords['Conduit'] == '-'.join(elem.split('-')[:2]),
                                                       ['XCoord', 'YCoord']].values[0])
            except:
                prod_wells_coords.append([None, None])
                continue
    prod_wells_coords_df = pd.DataFrame(prod_wells_coords)
    prod_wells_coords_df = (prod_wells_coords_df.ffill() + prod_wells_coords_df.bfill())/2
    inj_wells_coords = []
    for ind, elem in enumerate(inj_wells):
        try:
            inj_wells_coords.append(ws_coords2.loc[ws_coords2['Well Name'] == '-'.join(elem.split('-')[:2]),
                                                   ['X Coordinates of Extraction Point(UTM42)',
                                                    'Y Coordinates of Extraction Point(UTM42)']].values[0])
        except:
            try:
                inj_wells_coords.append(ws_coords.loc[ws_coords['Conduit'] == '-'.join(elem.split('-')[:2]),
                                                      ['XCoord', 'YCoord']].values[0])
            except:
                inj_wells_coords.append([None, None])
                continue
    inj_wells_coords_df = pd.DataFrame(inj_wells_coords)
    inj_wells_coords_df = (inj_wells_coords_df.ffill() + inj_wells_coords_df.bfill())/2
    return (x_all2, y_all2), (prod_wells_coords_df.values, inj_wells_coords_df.values)
