import xarray as xr
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ['CARTOPY_DATA_DIR'] = 'data/cartopy_data'
import itertools
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
import cartopy
cartopy.config['downloaders'] = {}
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
cartopy.config['pre_existing_data_dir'] = 'data/cartopy_data'
cartopy.config['data_dir'] = 'data/cartopy_data'
from cartopy.io.shapereader import Reader
import matplotlib.gridspec as gridspec

print(cartopy.config)
shapefile_path = 'data/cartopy_data/natural_earth/physical/ne_10m_coastline.shp'
coastline = cfeature.ShapelyFeature(
    Reader(shapefile_path).geometries(),
    ccrs.PlateCarree(),
    facecolor='none',
    edgecolor='black'
)

#list(cfeature.COASTLINE.geometries())

#print('OK: Coastline chargée localement')

import gc
import time
# --------------------------
# Configuration
# --------------------------
now = time.time()
data_path = 'data'
plot_dir = '../data/plots'
os.makedirs(plot_dir, exist_ok=True)

vars_dict = {'sfcWind': 'ff10', 'uas': 'u10', 'vas': 'v10', 'sfcWindmax': 'ff10max'}
variables = list(vars_dict.keys())

methods = ['GCM', 'CDF-t', 'Dual FM', 'Dual SDE', 'SI', 'SF 1200 km']
generative_methods = ['Dual SDE', 'SI', 'SF 1200 km']
#['GCM', 'CDF-t', 'R2D2', 'Dual FM', 'Dual SDE', 'SF 1200 km', 'SF 750 km', 'SF 300 km']

hist_period = slice('2000-01-01','2020-12-31')
fut_periods = [slice('2040-01-01','2060-12-31'), slice('2080-01-01','2100-12-31')]
fut_names = ['2040-2060','2080-2100']
n_members = 10
seasons = {'DJF':[12,1,2],'MAM':[3,4,5],'JJA':[6,7,8],'SON':[9,10,11]}

nlat, nlon = 45, 65
dx = 25  # km per grid cell
r_cut = [1, 2, 3] 
distance = {1:'1200 km', 2: '750 km', 3: '300 km'}
now = time.time()
out_mean = f'{data_path}/results/final_ds_mean.zarr'
out_members = f'{data_path}/results/final_ds_members.zarr'


def compute_nans(ds):
    for var in ds.data_vars:
        arr = ds[var]
        n_nans = arr.isnull().sum().compute()
        total = arr.size  # nombre total d’éléments
        pct = 100 * n_nans / total
        print(f'{var:20} : {n_nans:10} NaNs ({pct:5.2f}%)')

load_data = False
gcm_ds = xr.open_dataset(f'{data_path}/ACCESS_wind_FR_daily.nc')
times = gcm_ds.time
lat = gcm_ds.lat
lon = gcm_ds.lon
del gcm_ds
gc.collect()
if load_data:
    # --------------------------
    # Phase 1 : Préparer et sauvegarder les moyennes
    # --------------------------
    print('Loading GCM dataset (lazy)...')
    gcm_ds = xr.open_dataset(f'{data_path}/ACCESS_wind_FR_daily.nc')
    print('GCM dataset loaded.')

    print('Loading ERA5 dataset (lazy)...')
    rea_ds = xr.open_dataset(f'{data_path}/ERA5_wind_FR_daily.nc')
    print('ERA5 dataset loaded.')

    gcm_ds = gcm_ds.interp(lat=rea_ds.lat, lon=rea_ds.lon, kwargs={'fill_value': 'extrapolate'})
    gcm_ds = gcm_ds.sel(time=gcm_ds.time.dt.year>=1980)
    print('Loading stats methods dataset...')
    cdft_ds = xr.open_dataset(f'{data_path}/results/cdft_val.nc')
    r2d2_ds = xr.open_dataset(f'{data_path}/results/r2d2_val.nc')
    dotc_ds = xr.open_dataset(f'{data_path}/results/dotc_val.nc')
    print('Stats methods dataset loaded.')
    members = xr.Dataset()
    times = gcm_ds.time
    lat = gcm_ds.lat
    lon = gcm_ds.lon
    members = members.assign_coords(
                    time=times,
                    lat=lat,
                    lon=lon,
                    member=np.arange(n_members),
                )
    # --------------------------
    # Calcul des moyennes et std pour normalisation
    # --------------------------
    norm_stats = {}
    for v in variables:
        ref = vars_dict[v]
        norm_stats[v] = {
            'mean': rea_ds.sel(time=rea_ds.time.dt.year<2000)[ref].mean(dim='time').values,
            'std':  rea_ds.sel(time=rea_ds.time.dt.year<2000)[ref].std(dim='time').values
        }
    print(norm_stats)    
    print(f'ERA5 and stats methods to GCM')
    for v in variables:
        gcm_ds[f'CDF-t {v}'] = cdft_ds[v]
        gcm_ds[f'R2D2 {v}'] = r2d2_ds[v]
        gcm_ds[f'dOTC {v}'] = dotc_ds[v]
        gcm_ds[f'ERA5 {v}'] = rea_ds[vars_dict[v]]
        gcm_ds = gcm_ds.rename({v: f'GCM {v}'})

    compute_nans(gcm_ds)

    gcm_ds.to_zarr(out_mean, mode='w',zarr_format=2)
    compute_nans(gcm_ds)

    del cdft_ds, r2d2_ds, rea_ds, gcm_ds, dotc_ds
    gc.collect()
    # --------------------------
    # SF mean + members
    # --------------------------
    
    for r in r_cut:
        print(f'Processing SF r={r}')
        preds = []
        for m in range(n_members):
            p = torch.load(f'{data_path}/results/ACCESS_to_ERA5_all_wind_{r}_{m}.pt')
            for i, v in enumerate(variables):
                p_mean = p[:, i].mean(dim=0)
                p_std = p[:, i].std(dim=0)
                p[:, i] = ((p[:, i]-p_mean)/p_std * norm_stats[v]['std'] + norm_stats[v]['mean']).to(torch.float32)
                #print(f'        v = {v}, m = {m} mean = {p[:, i].mean().item():.2f}, std = {p[:, i].std().item():.2f}')
            print(f'  Loading SF r={r}, m={m}, shape={p.shape}')
            preds.append(p)
            del p
            gc.collect()
        preds = np.stack(preds, axis=0)  # (member, time, var)
        means = preds.mean(axis=0)
        if members is None:
            members = xr.open_zarr(out_members, consolidated=True)
        for i, v in enumerate(variables):
            #print(f'SF {distance[r]}, v={v}, mean={means[:, i].mean():.2f}, std={means[:, i].std():.2f}')
            members[f'SF {distance[r]} {v}'] = xr.DataArray(
                preds[:, :, i],
                dims=('member', 'time', 'lat', 'lon'),
                coords={
                    'member': members.member,
                    'time': members.time,
                    'lat': members.lat,
                    'lon': members.lon,
                },
            )
        del preds
        gc.collect()
        print('\nSaving to Zarr')
        if r == r_cut[0]:
            members.to_zarr(out_members,mode='w',zarr_format=2)
        else:
            members.to_zarr(out_members, mode='a')
        compute_nans(members)
        del members
        members = None
        gc.collect()

        rea = xr.open_zarr(out_mean, consolidated=True)
        for v in variables:
            rea[f'SF {distance[r]} {v}'] = xr.DataArray(
                means[:, variables.index(v)].reshape(-1, nlat, nlon),
                dims=('time', 'lat', 'lon'),
                coords={
                    'time': rea.time,
                    'lat': rea.lat,
                    'lon': rea.lon,
                },
            )
        rea.to_zarr(out_mean, mode='a')
        del means, rea
    # --------------------------
    # Dual FM mean + members
    # --------------------------
    print(f'Processing Dual FM')
    preds = []
    for m in range(n_members):
        p = torch.load(f'{data_path}/results/ACCESS_to_ERA5_dual_fm_{m}.pt')
        for i, v in enumerate(variables):
            p_mean = p[:, i].mean(dim=0)
            p_std = p[:, i].std(dim=0)
            p[:, i] = ((p[:, i]-p_mean)/p_std * norm_stats[v]['std'] + norm_stats[v]['mean']).to(torch.float32)
            #print(f'        v = {v}, m = {m} mean = {p[:, i].mean().item():.2f}, std = {p[:, i].std().item():.2f}')
        print(f'  Loading Dual FM, m={m}, shape={p.shape}')
        preds.append(p)
        del p
        gc.collect()
    preds = np.stack(preds, axis=0)  # (member, time, var)
    means = preds.mean(axis=0)
    members = xr.open_zarr(out_members, consolidated=True)
    for i, v in enumerate(variables):
        #print(f'Dual FM, v={v}, mean={means[:, i].mean():.2f}, std={means[:, i].std():.2f}')
        members[f'Dual FM {v}'] = xr.DataArray(
                preds[:, :, i],
                dims=('member', 'time', 'lat', 'lon'),
                coords={
                    'member': members.member,
                    'time': members.time,
                    'lat': members.lat,
                    'lon': members.lon,
                },
            )
    del preds
    gc.collect()
    print('\nSaving to Zarr')
    members.to_zarr(out_members, mode='a')
    compute_nans(members)
    del members
    gc.collect()
    
    rea = xr.open_zarr(out_mean, consolidated=True)
    for v in variables:
        rea[f'Dual FM {v}'] = xr.DataArray(
            means[:, variables.index(v)].reshape(-1, nlat, nlon),
            dims=('time', 'lat', 'lon'),
            coords={
                'time': rea.time,
                'lat': rea.lat,
                'lon': rea.lon,
            },
        )
    rea.to_zarr(out_mean, mode='a')
    del means, rea
    gc.collect()
    print(f'Data preparation completed in {time.time() - now:.2f} seconds.')
    print(f'Processing Dual SDE')
    preds = []
    for m in range(n_members):
        p = torch.load(f'{data_path}/results/ACCESS_to_ERA5_dual_fm_sde_{m}.pt')
        for i, v in enumerate(variables):
            p_mean = p[:, i].mean(dim=0)
            p_std = p[:, i].std(dim=0)
            p[:, i] = ((p[:, i]-p_mean)/p_std * norm_stats[v]['std'] + norm_stats[v]['mean']).to(torch.float32)
        print(f'  Loading Dual SDE, m={m}, shape={p.shape}')
        preds.append(p)
        del p
        gc.collect()
    preds = np.stack(preds, axis=0)  # (member, time, var)
    means = preds.mean(axis=0)
    members = xr.open_zarr(out_members, consolidated=True)
    for i, v in enumerate(variables):
        members[f'Dual SDE {v}'] = xr.DataArray(
                preds[:, :, i],
                dims=('member', 'time', 'lat', 'lon'),
                coords={
                    'member': members.member,
                    'time': members.time,
                    'lat': members.lat,
                    'lon': members.lon,
                },
                )
    del preds
    gc.collect()
    print('\nSaving to Zarr')
    members.to_zarr(out_members, mode='a')
    compute_nans(members)
    del members
    gc.collect()    
    rea = xr.open_zarr(out_mean, consolidated=True)
    for v in variables:
        rea[f'Dual SDE {v}'] = xr.DataArray(
            means[:, variables.index(v)].reshape(-1, nlat, nlon),
                dims=('time', 'lat', 'lon'),
                coords={
                    'time': rea.time,
                    'lat': rea.lat,
                    'lon': rea.lon,
                },
        )
    rea.to_zarr(out_mean, mode='a')
    del means, rea

    print(f'Processing SI')
    preds = []
    for m in range(n_members):
        p = torch.load(f'{data_path}/results/ACCESS_to_ERA5_dual_fm_si_{m}.pt')
        for i, v in enumerate(variables):
            p_mean = p[:, i].mean(dim=0)
            p_std = p[:, i].std(dim=0)
            p[:, i] = ((p[:, i]-p_mean)/p_std * norm_stats[v]['std'] + norm_stats[v]['mean']).to(torch.float32)
        print(f'  Loading SI, m={m}, shape={p.shape}')
        preds.append(p)
        del p
        gc.collect()
    preds = np.stack(preds, axis=0)  # (member, time, var)
    means = preds.mean(axis=0)
    members = xr.open_zarr(out_members, consolidated=True)
    for i, v in enumerate(variables):
        members[f'SI {v}'] = xr.DataArray(
                preds[:, :, i],
                dims=('member', 'time', 'lat', 'lon'),
                coords={
                    'member': members.member,
                    'time': members.time,
                    'lat': members.lat,
                    'lon': members.lon,
                },
                )
    del preds
    gc.collect()
    print('\nSaving to Zarr')
    members.to_zarr(out_members, mode='a')
    compute_nans(members)
    del members
    gc.collect()    
    rea = xr.open_zarr(out_mean, consolidated=True)
    for v in variables:
        rea[f'SI {v}'] = xr.DataArray(
            means[:, variables.index(v)].reshape(-1, nlat, nlon),
                dims=('time', 'lat', 'lon'),
                coords={
                    'time': rea.time,
                    'lat': rea.lat,
                    'lon': rea.lon,
                },
        )
    rea.to_zarr(out_mean, mode='a')
    del means, rea
gc.collect()

now = time.time()
# --------------------------
# Utility functions
# --------------------------

def plot_spatial_map(
    data,
    title,
    filename,
    diverging=True,
    center=0.0,
    vmin=None,
    vmax=None,
    cmap_div='BrBG',
    cmap_seq='cividis',
):
    # --- Ensure DataArray ---
    if isinstance(data, np.ndarray):
        data = xr.DataArray(data)

    # --- Reduce to 2D if needed ---
    if data.ndim == 3:
        data = data.mean(dim=data.dims[0])

    # --- Abort if fully NaN ---
    if data.isnull().all():
        print(f'⚠️ plot_spatial_map skipped (all-NaN): {title}')
        return

    # --- Select colormap ---
    cmap = cmap_div if diverging else cmap_seq

    values = data.values

    # --- Automatic vmin / vmax ---
    if vmin is None or vmax is None:
        if diverging:
            # Remove NaNs before computing scale
            diff = values - center
            if np.all(np.isnan(diff)):
                print(f'⚠️ plot_spatial_map skipped (NaN after centering): {title}')
                return

            max_val = np.nanmax(np.abs(diff))

            # Handle constant fields
            if max_val == 0 or not np.isfinite(max_val):
                max_val = 1e-6

            vmin = center - max_val
            vmax = center + max_val

        else:
            if np.all(np.isnan(values)):
                print(f'⚠️ plot_spatial_map skipped (NaN values): {title}')
                return

            vmin = np.nanmin(values)
            vmax = np.nanmax(values)

            # Handle constant fields
            if vmin == vmax or not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin -= 1e-6
                vmax += 1e-6

    # --- Plot with Cartopy ---
    fig = plt.figure(figsize=(6, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --- Auto extent from xarray ---
    lon_name = [d for d in data.dims if 'lon' in d.lower()][0]
    lat_name = [d for d in data.dims if 'lat' in d.lower()][0]

    lon_min = float(data[lon_name].min())
    lon_max = float(data[lon_name].max())
    lat_min = float(data[lat_name].min())
    lat_max = float(data[lat_name].max())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    im = data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
    )

    # Côtes et frontières
    coastline = cfeature.ShapelyFeature(
        Reader(shapefile_path).geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black'
    )

    ax.add_feature(coastline)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def radar_plot(
    metrics_dict,
    methods_list,
    categories,
    title,
    fname,
    metric_direction,
    normalize=True
):
    '''
    Radar plot avec normalisation par métrique + affichage des ranges physiques.

    Parameters
    ----------
    metrics_dict : dict
        Clés du type '{method}_{metric}'
    methods_list : list
        Méthodes à comparer
    categories : list
        Liste des métriques (ex: ['delta_full','corr','anom'])
    metric_direction : dict
        {'metric': 'min' ou 'max'}
        'min' = plus petit est meilleur
        'max' = plus grand est meilleur
    normalize : bool
        Normalisation sur [0,1] par métrique
    '''

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # -----------------------------
    # Collecte des valeurs par métrique
    # -----------------------------
    cat_vals = {cat: [] for cat in categories}
    for cat in categories:
        for method in methods_list:
            cat_vals[cat].append(metrics_dict[f'{method}_{cat}'])

    cat_min = {cat: np.min(cat_vals[cat])-0.1*np.min(cat_vals[cat]) for cat in categories}
    cat_max = {cat: np.max(cat_vals[cat])+0.1*np.min(cat_vals[cat]) for cat in categories}

    # -----------------------------
    # Préparer labels avec ranges physiques
    # -----------------------------
    labels = []
    for cat in categories:
        mn = cat_min[cat]
        mx = cat_max[cat]

        if metric_direction[cat] == 'min':
            label = f'{cat}' #\n({mn:.2g} → {mx:.2g})'
        else:
            label = f'{cat}' #\n({mx:.2g} → {mn:.2g})'

        labels.append(label)

    # -----------------------------
    # Normalisation
    # -----------------------------
    plot_values = {}
    for method in methods_list:
        vals = []
        for cat in categories:
            val = metrics_dict[f'{method}_{cat}']

            if normalize:
                if cat_max[cat] - cat_min[cat] == 0:
                    val_norm = 0.5
                else:
                    if metric_direction[cat] == 'min':
                        # plus petit = mieux
                        val_norm = (cat_max[cat] - val) / (cat_max[cat] - cat_min[cat])
                    else:
                        # plus grand = mieux
                        val_norm = (val - cat_min[cat]) / (cat_max[cat] - cat_min[cat])
                vals.append(val_norm)
            else:
                vals.append(val)

        vals += vals[:1]
        plot_values[method] = vals

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for method, vals in plot_values.items():
        ax.plot(angles, vals, linewidth=2, label=method)
        ax.fill(angles, vals, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, fontsize=16)

    if normalize:
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

def compute_global_mean_abs_diff(a,b):
    a = a.mean(dim='time')
    b = b.mean(dim='time')
    return np.nanmean(np.abs(a.values - b.values))

def compute_global_std_diff(a,b):
    a = a.std(dim='time')
    b = b.std(dim='time')
    return np.nanmean(np.abs(a.values - b.values))

def compute_ks_stat(obs, method=None, members=None, quantile=None):
    obs_vals = obs.flatten()[~np.isnan(obs.flatten())]    
    if members is not None:
        method_vals = members.flatten()
    else:
        method_vals = method.flatten()
    
    method_vals = method_vals[~np.isnan(method_vals)]
    
    if quantile is None:
        return ks_2samp(obs_vals, method_vals)[0]
    
    thr = np.quantile(obs_vals, quantile)
    if quantile < 0.5:
        obs_tail = obs_vals[obs_vals <= thr]
        method_tail = method_vals[method_vals <= thr]
    else:
        obs_tail = obs_vals[obs_vals >= thr]
        method_tail = method_vals[method_vals >= thr]
    
    return ks_2samp(obs_tail, method_tail)[0]

def plot_cdf_subplot(rea, v, methods, mask=None, name='global', members=None):
    '''
    Plot CDFs en 3 subplots : full, quantiles>0.9, quantiles<0.1.
    Affiche aussi les membres avec alpha faible.
    '''
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    # --- ERA5 ---
    era5 = rea[f'ERA5 {v}']
    if mask is not None:
        era5 = era5.where(mask)
    era5_vals = np.sort(era5.values.flatten())
    era5_vals = era5_vals[~np.isnan(era5_vals)]
    cdf_era5 = np.linspace(0, 1, len(era5_vals))

    axes[0].plot(era5_vals, cdf_era5, label='ERA5', color='k')
    axes[1].plot(era5_vals[cdf_era5>0.9], cdf_era5[cdf_era5>0.9], label='ERA5', color='k')
    axes[2].plot(era5_vals[cdf_era5<0.1], cdf_era5[cdf_era5<0.1], label='ERA5', color='k')

    # --- Préparer les méthodes ---
    colors = plt.get_cmap('tab10', len(methods))
    for idx, m in enumerate(methods):
        da = rea[f'{m} {v}']
        if mask is not None:
            da = da.where(mask)
        vals = np.sort(da.values.flatten())
        vals = vals[~np.isnan(vals)]
        cdf_vals = np.linspace(0,1,len(vals))

        # Moyenne
        for ax, qmask in zip(axes, [slice(None), cdf_vals>0.9, cdf_vals<0.1]):
            ax.plot(vals[qmask], cdf_vals[qmask], label=m, color=colors(idx))

        # Membres
        if members is not None:
            if f'{m} {v}' in members.data_vars:
                member_vars = members[f'{m} {v}']
                for i in range(10):
                    mda = member_vars.sel(member=i)
                    mvals = np.sort(mda.values.flatten())
                    mvals = mvals[~np.isnan(mvals)]
                    mcdf = np.linspace(0,1,len(mvals))
                    for ax, qmask in zip(axes, [slice(None), mcdf>0.9, mcdf<0.1]):
                        ax.plot(mvals[qmask], mcdf[qmask], color=colors(idx), alpha=0.3, zorder=1)
    for ax in axes:
        ax.set_xlabel(v)
        ax.set_ylabel('CDF')
    axes[0].set_title('Full CDF')
    axes[1].set_title('Quantiles > 0.9')
    axes[2].set_title('Quantiles < 0.1')
    
    # Légende centrée sous les subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=len(methods)+1, frameon=False)

    plt.suptitle(f'CDF {v} – {name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Faire de la place pour la légende
    plt.savefig(f'{plot_dir}/cdf_{v}_{name}_subplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def compute_delta_metric(hist_data, fut_data, gcm_hist, gcm_fut, months=None):
    if months is not None:
        hist_data = hist_data.sel(time=hist_data.time.dt.month.isin(months))
        fut_data = fut_data.sel(time=fut_data.time.dt.month.isin(months))
        gcm_hist = gcm_hist.sel(time=gcm_hist.time.dt.month.isin(months))
        gcm_fut = gcm_fut.sel(time=gcm_fut.time.dt.month.isin(months))

    if hist_data.time.size == 0 or fut_data.time.size == 0:
        return np.nan, np.nan, np.nan

    hist_mean = hist_data.mean(dim='time')
    fut_mean = fut_data.mean(dim='time')
    gcm_hist_mean = gcm_hist.mean(dim='time')
    gcm_fut_mean = gcm_fut.mean(dim='time')

    delta = xr.where(hist_mean != 0, (fut_mean - hist_mean) / hist_mean * 100, np.nan)
    gcm_delta = xr.where(gcm_hist_mean != 0, (gcm_fut_mean - gcm_hist_mean) / gcm_hist_mean * 100, np.nan)

    score = np.abs(delta - gcm_delta).mean(skipna=True).compute().item()

    return score, delta, gcm_delta

def plot_delta_variable_grid(
    gcm,
    variables,
    methods,          
    fut_periods,
    fut_names,
    hist_period,
    seasons,
    plot_dir,
    shapefile_path=shapefile_path,
    seasonal=False
):
    all_methods = ['GCM'] + methods

    for v in variables:

        # -------- rows definition + labels
        rows = []
        row_labels = []
        if seasonal:
            for fut_idx, fut_name in enumerate(fut_names):
                for season, months in seasons.items():
                    rows.append((fut_name, season))
                    start, end = fut_periods[fut_idx].start, fut_periods[fut_idx].stop
                    row_labels.append(f'{fut_name} – {season}')
        else:
            for fut_idx, fut_name in enumerate(fut_names):
                rows.append((fut_name, 'FULL'))
                start, end = fut_periods[fut_idx].start, fut_periods[fut_idx].stop
                row_labels.append(f'{fut_name}')

        nrows = len(rows)
        ncols = len(all_methods)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4*ncols, 3.5*nrows),
            constrained_layout=False,
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        if nrows == 1:
            axes = axes[np.newaxis, :]
        if ncols == 1:
            axes = axes[:, np.newaxis]

        fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.05, wspace=0.1, hspace=0.15)

        # -------- compute all deltas first (common colorbar)
        delta_maps = {}
        for fut_idx, fut_period in enumerate(fut_periods):
            fut_name = fut_names[fut_idx]

            for season, months in ([('FULL', None)] if not seasonal else seasons.items()):
                for method in all_methods:
                    hist_data = gcm[f'{method} {v}'].sel(time=hist_period)
                    fut_data  = gcm[f'{method} {v}'].sel(time=fut_period)

                    gcm_hist = gcm[f'GCM {v}'].sel(time=hist_period)
                    gcm_fut  = gcm[f'GCM {v}'].sel(time=fut_period)

                    _, delta, _ = compute_delta_metric(
                        hist_data, fut_data,
                        gcm_hist, gcm_fut,
                        months
                    )
                    delta_maps[(fut_name, season, method)] = delta

        # -------- common color scale
        stack = np.stack([d.values for d in delta_maps.values()])
        vmax = np.nanmax(np.abs(stack))
        vmin = -vmax

        # -------- plotting
        for i, (fut_name, season) in enumerate(rows):
            for j, method in enumerate(all_methods):
                ax = axes[i, j]
                delta = delta_maps[(fut_name, season, method)]

                # Detect lon/lat dimensions
                lon_name = [d for d in delta.dims if 'lon' in d.lower()][0]
                lat_name = [d for d in delta.dims if 'lat' in d.lower()][0]

                lon = delta[lon_name].values
                lat = delta[lat_name].values

                # Plot with pcolormesh
                im = ax.pcolormesh(
                    lon, lat, delta,
                    vmin=vmin, vmax=vmax,
                    cmap='BrBG',
                    transform=ccrs.PlateCarree()
                )

                # Contours
                coastline = cfeature.ShapelyFeature(
                    Reader(shapefile_path).geometries(),
                    ccrs.PlateCarree(),
                    facecolor='none',
                    edgecolor='black'
                )
                ax.add_feature(coastline)

                # Automatic extent
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

                # Title only on top row
                if i == 0:
                    ax.set_title(method, fontsize=12)

                # Row labels on first column
                if j == 0:
                    ax.text(
                        -0.05, 0.5, row_labels[i],
                        transform=ax.transAxes,
                        fontsize=11,
                        rotation=90,
                        va='center',
                        ha='right'
                    )

        # -------- colorbar
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label('Relative change (%)')

        title = f'Relative delta – {v}'
        if seasonal:
            title += ' (seasonal)'
        fig.suptitle(title, fontsize=15)

        suffix = 'seasonal' if seasonal else 'full'
        plt.savefig(f'{plot_dir}/delta_{suffix}_{v}.png', dpi=300)
        plt.close()

def intervar_corr_matrix(rea, method):
    mat = np.zeros((len(variables), len(variables)))
    for i,v1 in enumerate(variables):
        for j,v2 in enumerate(variables):
            mat[i,j] = xr.corr(
                rea[f'{method} {v1}'],
                rea[f'{method} {v2}'],
                dim='time'
            ).mean().compute().item()
    return pd.DataFrame(mat, index=variables, columns=variables)

def plot_seasonal_cycle(rea, variables, methods, plot_dir, members):
    '''
    Plot seasonal cycle (monthly mean) for ERA5 + all methods on the same plot for each variable.
    Members are plotted in faint color (alpha=0.2) for each method.
    Adapted for members with dimensions (member, time, lat, lon)
    '''
    if members is not None:
        print(f'MEMBERS VARS: {members.data_vars}')
    
    for v in variables:
        plt.figure(figsize=(8,5))
        
        # --- ERA5 ---
        era5_cycle = rea[f'ERA5 {v}'].groupby('time.month').mean(dim=['time','lat','lon'])
        months = era5_cycle.month
        plt.plot(months, era5_cycle, label='ERA5', color='k', lw=1)
        
        # --- Méthodes principales ---
        colors = plt.get_cmap('tab10', len(methods))
        for idx, method in enumerate(methods):
            model_cycle = rea[f'{method} {v}'].groupby('time.month').mean(dim=['time','lat','lon'])
            plt.plot(months, model_cycle, label=method, color=colors(idx), lw=1)
            
            # --- Membres ---
            if members is not None:
                if f'{method} {v}' in members.data_vars:                
                    member_vars = members[f'{method} {v}'] # shape = (member, time, lat, lon)
                    mcycle = member_vars.groupby('time.month').mean(dim=['time','lat','lon'])  # shape = (member, month)                    
                    for m_idx in range(mcycle.sizes['member']):
                        plt.plot(
                            months,
                            mcycle.isel(member=m_idx),
                            color=colors(idx),
                            alpha=0.4,
                            zorder=1
                        )
        
        plt.title(f'Mean Seasonal Cycle – {v}')
        plt.xlabel('Month')
        plt.ylabel(v)
        plt.xticks(months)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/seasonal_cycle_{v}_all_methods.png', dpi=300)
        plt.close()

def plot_temporal_correlation_subplot(gcm, variables, methods, plot_dir):

    n_vars = len(variables)
    n_methods = len(methods)

    fig = plt.figure(
        figsize=(5*n_methods + 0.8, 4*n_vars)
    )

    gs = gridspec.GridSpec(
        n_vars,
        n_methods + 1,
        width_ratios=[1]*n_methods + [0.04],
        wspace=0.05,
        hspace=0.15
    )

    axes = np.empty((n_vars, n_methods), dtype=object)

    for i, v in enumerate(variables):
        for j, method in enumerate(methods):

            ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
            axes[i, j] = ax

            corr_map = xr.corr(
                gcm[f'{method} {v}'],
                gcm[f'GCM {v}'],
                dim='time'
            )

            lon_name = [d for d in corr_map.dims if 'lon' in d.lower()][0]
            lat_name = [d for d in corr_map.dims if 'lat' in d.lower()][0]

            lon = corr_map[lon_name].values
            lat = corr_map[lat_name].values

            im = ax.pcolormesh(
                lon, lat, corr_map,
                vmin=-1, vmax=1,
                cmap='BrBG',
                transform=ccrs.PlateCarree()
            )

            coastline = cfeature.ShapelyFeature(
                Reader(shapefile_path).geometries(),
                ccrs.PlateCarree(),
                facecolor='none',
                edgecolor='black'
            )
            ax.add_feature(coastline)

            ax.set_extent(
                [lon.min(), lon.max(), lat.min(), lat.max()],
                crs=ccrs.PlateCarree()
            )

            if i == 0:
                ax.set_title(method, fontsize=12)

            if j == 0:
                ax.text(
                    -0.15, 0.5, v,
                    transform=ax.transAxes,
                    fontsize=14, fontweight='bold',
                    va='center', ha='right', rotation=90
                )

    # --- Axe dédié pour la colorbar ---
    cax = fig.add_subplot(gs[:, -1])

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Temporal correlation')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    fig.suptitle('Temporal correlation vs GCM', fontsize=16)
    plt.savefig(
        f'{plot_dir}/temporal_corr_all_methods.png',
        dpi=300
    )
    plt.close()

def plot_mean_std_diffs(rea, variables, methods, plot_dir):
    '''
    Compute mean and std deviations differences per grid point
    and plot boxplots for each variable comparing methods.
    
    Parameters
    ----------
    rea : xr.Dataset
    variables : list of str
    methods : list of str
    plot_dir : str
    '''
    metrics_vs_era5 = {}

    for v in variables:
        mean_diffs_all = []
        std_diffs_all = []

        for method in methods:
            data_model = rea[f'{method} {v}']
            data_era5 = rea[f'ERA5 {v}']

            # Moyenne et std par grille
            mean_model = data_model.mean(dim='time')
            mean_era5 = data_era5.mean(dim='time')
            std_model = data_model.std(dim='time')
            std_era5 = data_era5.std(dim='time')

            # Différence absolue par grille
            mean_diff_grid = np.abs(mean_model - mean_era5).values.flatten()
            std_diff_grid = np.abs(std_model - std_era5).values.flatten()

            mean_diffs_all.append(mean_diff_grid)
            std_diffs_all.append(std_diff_grid)
            
        # Boxplot
        plt.figure(figsize=(10,5))
        positions = np.arange(len(methods))
        # Moyennes
        plt.boxplot(mean_diffs_all, positions=positions-0.2, widths=0.35, patch_artist=True,
                    boxprops=dict(facecolor='#4C72B0'), tick_labels=methods)
        # Std
        plt.boxplot(std_diffs_all, positions=positions+0.2, widths=0.35, patch_artist=True,
                    boxprops=dict(facecolor='#DD8452'), tick_labels=methods)
        plt.xticks(positions, methods)
        plt.ylabel(f'Difference in {v}')
        plt.title(f'Grid-point differences of mean (blue) and std (red) – {v}')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/mean_std_boxplot_{v}.png', dpi=300)
        plt.close()

    return metrics_vs_era5

def compute_mean_std_diffs(rea, variables, methods, plot_dir):
    '''
    Compute per-grid differences of mean and std against ERA5,
    plot boxplots, and return metrics to use in radar plots.
    
    Parameters
    ----------
    rea : xr.Dataset
        Dataset with ERA5 and model data.
    variables : list of str
        Variable names.
    methods : list of str
        Model/method names.
    plot_dir : str
        Folder to save plots.
        
    Returns
    -------
    metrics_vs_era5 : dict
        Dict containing mean and std metrics for radar plots.
        Keys like '{method}_mean_diff' and '{method}_std_diff'.
    '''
    metrics_vs_era5 = {}

    for v in variables:
        mean_diffs_all = []
        std_diffs_all = []

        for method in methods:
            model = rea[f'{method} {v}']
            era5 = rea[f'ERA5 {v}']

            # Moyenne et std par grille
            mean_model = model.mean(dim='time')
            mean_era5 = era5.mean(dim='time')
            std_model = model.std(dim='time')
            std_era5 = era5.std(dim='time')

            # Différence absolue par grille
            mean_diff_grid = np.abs(mean_model - mean_era5).values.flatten()
            std_diff_grid = np.abs(std_model - std_era5).values.flatten()

            mean_diffs_all.append(mean_diff_grid)
            std_diffs_all.append(std_diff_grid)

            # Moyenne globale pour radar plot
            metrics_vs_era5[f'{method}_{v}_mean_diff'] = np.nanmean(mean_diff_grid)
            metrics_vs_era5[f'{method}_{v}_std_diff'] = np.nanmean(std_diff_grid)

        # Boxplot par variable
        plt.figure(figsize=(10,5))
        positions = np.arange(len(methods))
        # Moyennes
        plt.boxplot(mean_diffs_all, positions=positions-0.2, widths=0.35, patch_artist=True,
                    boxprops=dict(facecolor='#4C72B0'), tick_labels=methods)
        # Std
        plt.boxplot(std_diffs_all, positions=positions+0.2, widths=0.35, patch_artist=True,
                    boxprops=dict(facecolor='#DD8452'), tick_labels=methods)
        plt.xticks(positions, methods)
        plt.ylabel(f'Difference in {v}')
        plt.title(f'Grid-point differences of mean (blue) and std (red) – {v}')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/mean_std_boxplot_{v}.png', dpi=300)
        plt.close()

    # Pour radar plots, on peut aussi calculer moyenne par méthode sur toutes les variables
    for method in methods:
        mean_vals = [metrics_vs_era5[f'{method}_{v}_mean_diff'] for v in variables]
        std_vals = [metrics_vs_era5[f'{method}_{v}_std_diff'] for v in variables]
        metrics_vs_era5[f'{method}_mean_overall'] = np.mean(mean_vals)
        metrics_vs_era5[f'{method}_std_overall'] = np.mean(std_vals)

    return metrics_vs_era5

def plot_intervar_corr_grid(
    rea,
    variables,
    methods,
    plot_dir,
    corr_type='spearman',
    shapefile_path=shapefile_path  # chemin vers le shapefile local
):
    '''
    Plot spatial inter-variable correlations per grid point using Cartopy.
    One figure per variable, with all methods and correlations with other variables.

    Layout:
        rows    = methods (ERA5 + methods)
        columns = other variables
    '''

    all_methods = ['ERA5'] + methods

    for v in variables:
        other_vars = [u for u in variables if u != v]

        nrows = len(all_methods)
        ncols = len(other_vars)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        for i, method in enumerate(all_methods):
            for j, u in enumerate(other_vars):
                ax = axes[i, j]

                da1 = rea[f'{method} {v}']
                da2 = rea[f'{method} {u}']

                # Compute correlation
                if corr_type.lower() == 'pearson':
                    corr = xr.corr(da1, da2, dim='time')
                elif corr_type.lower() == 'spearman':
                    corr = xr.corr(
                        da1.rank(dim='time'),
                        da2.rank(dim='time'),
                        dim='time'
                    )
                else:
                    raise ValueError('corr_type must be pearson or spearman')

                # Detect lon/lat dims
                lon_name = [d for d in corr.dims if 'lon' in d.lower()][0]
                lat_name = [d for d in corr.dims if 'lat' in d.lower()][0]

                lon = corr[lon_name].values
                lat = corr[lat_name].values

                # Plot with Cartopy
                im = ax.pcolormesh(
                    lon, lat, corr,
                    vmin=-1, vmax=1,
                    cmap='BrBG',
                    transform=ccrs.PlateCarree()
                )

                # Ajouter les contours / borders
                if shapefile_path is not None:
                    from cartopy.io.shapereader import Reader
                    coastline = cfeature.ShapelyFeature(
                        Reader(shapefile_path).geometries(),
                        ccrs.PlateCarree(),
                        facecolor='none',
                        edgecolor='black',
                        linewidth=0.5
                    )
                    ax.add_feature(coastline)
                else:
                    # Fallback si pas de shapefile
                    ax.coastlines(linewidth=0.5)

                # Ajouter les lignes de grille
                ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

                # Titre de chaque subplot : méthode + variable
                ax.set_title(f'{method} - {v} vs {u}', fontsize=10)

                # Extent (optionnel, ajustez selon votre région)
                # ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

        # Ajouter une colorbar commune
        fig.subplots_adjust(right=0.92, hspace=0.25, wspace=0.25, top=0.95)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.set_label(f'{corr_type.capitalize()} Correlation', fontsize=12)

        # Titre général (plus proche des figures)
        fig.suptitle(
            f'Inter-variable Correlations for {v} ({corr_type.capitalize()})',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        # Sauvegarder
        plt.savefig(
            f'{plot_dir}/intervar_corr_{v}_{corr_type}.png',
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()
        
        print(f'Saved: intervar_corr_{v}_{corr_type}.png')

def compute_grid_distances(lat, lon):
    '''
    Compute pairwise distances between grid points in km.
    '''
    R = 6371.0  # Earth radius (km)

    lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')
    coords = np.column_stack([lat2d.ravel(), lon2d.ravel()])

    def haversine_vec(p1, p2):
        lat1, lon1 = np.radians(p1)
        lat2, lon2 = np.radians(p2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    dist = squareform(pdist(coords, metric=haversine_vec))
    return dist

def spatial_spearman_distance_and_metric(rea, variables, methods, lat, lon, members=None, nbins=40, short_dist_weight=1000.0):
    dist_matrix = compute_grid_distances(lat, lon)
    bins = np.linspace(0, np.nanmax(dist_matrix), nbins)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    weights = np.ones_like(bin_centers)
    weights[bin_centers <= short_dist_weight] = 2.0  # double weight for distances <= threshold
    weights = weights / np.sum(weights)               # normalize so sum = 1

    corr_vs_dist_dict = {}
    metrics_dict = {}

    # ---------------- ERA5 (reference) ----------------
    corr_era5_matrices = {}
    for v in variables:
        data = rea[f'ERA5 {v}'].values  # (time, lat, lon)
        nt, nlat, nlon = data.shape

        # Compute Spearman correlations (vectorized ranking)
        anom = data - data.mean(axis=(1,2), keepdims=True)
        anom = anom.reshape(nt, -1)  # (time, npoints)
        ranked = np.argsort(np.argsort(anom, axis=0), axis=0)
        corr_era5 = np.corrcoef(ranked.T)
        corr_era5_matrices[v] = corr_era5

        # Correlation vs distance bins
        corr_vs_dist_dict[(v, 'ERA5')] = np.array([
            np.nanmean(corr_era5[(dist_matrix >= bins[i]) & (dist_matrix < bins[i+1])])
            for i in range(len(bins)-1)
        ])

    # ---------------- METHODS ----------------
    for m in methods:
        metric_vals = []
        for v in variables:
            data = rea[f'{m} {v}'].values
            nt, nlat, nlon = data.shape

            anom = data - data.mean(axis=(1,2), keepdims=True)
            anom = anom.reshape(nt, -1)
            ranked = np.argsort(np.argsort(anom, axis=0), axis=0)
            corr_matrix = np.corrcoef(ranked.T)

            corr_vs_dist_dict[(v, m)] = np.array([
                np.nanmean(corr_matrix[(dist_matrix >= bins[i]) & (dist_matrix < bins[i+1])])
                for i in range(len(bins)-1)
            ])
            metric_vals.append(np.sum(np.abs(corr_matrix - corr_era5_matrices[v]) * weights[:, np.newaxis].mean()))

            if members is not None:
                member_corrs = []
                if f'{m} {v}' in members.data_vars:
                    mem_data = members[f'{m} {v}']
                    for i in range(10):
                        mem_data = mem_data.sel(member=i).values
                        anom_mem = mem_data - mem_data.mean(axis=(1,2), keepdims=True)
                        anom_mem = anom_mem.reshape(anom_mem.shape[0], -1)
                        ranked_mem = np.argsort(np.argsort(anom_mem, axis=0), axis=0)
                        corr_mem = np.corrcoef(ranked_mem.T)
                        member_corrs.append(corr_mem)
                        corr_vs_dist_dict[f'{v}_{m}_members_{i}'] = member_corrs

        metrics_dict[f'{m}_spatial_corr'] = np.mean(metric_vals)

    return corr_vs_dist_dict, metrics_dict, bins, dist_matrix

def plot_spatial_spearman_means(
    corr_vs_dist_dict, bins, variables, methods, plot_dir='../data/plots'
):
    colors = plt.get_cmap('tab10', len(methods))

    for v in variables:
        plt.figure(figsize=(7,5))

        # ERA5
        plt.plot(
            0.5*(bins[:-1]+bins[1:]),
            corr_vs_dist_dict[(v, 'ERA5')],
            color='k', label='ERA5', zorder=3
        )

        for idx, m in enumerate(methods):
            plt.plot(
                0.5*(bins[:-1]+bins[1:]),
                corr_vs_dist_dict[(v, m)],
                color=colors(idx),
                label=m
            )

        plt.axhline(0, color='k', lw=0.5)
        plt.grid(alpha=0.3)
        plt.ylim(-0.35, 1.05)

        plt.xlabel('Distance (km)')
        plt.ylabel('Spearman correlation')
        plt.title(f'Spatial Spearman correlation vs distance – {v}')
        plt.legend(frameon=False, ncol=3)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/spatial_spearman_distance_means_{v}.png', dpi=300)
        plt.show()

def plot_spatial_spearman_members(
    corr_vs_dist_dict,
    dist_matrix,
    bins,
    variables,
    generative_methods,
    members,
    plot_dir='../data/plots'
):
    colors = plt.get_cmap('tab10', len(generative_methods))

    for v in variables:
        plt.figure(figsize=(7,5))

        for idx, m in enumerate(generative_methods):
            color = colors(idx)
            if f'{m} {v}' in members.data_vars:
                member_vars = members[f'{m} {v}']
                for i in range(10):
                    memb_data = member_vars.sel(member=i).values
                    nt, nlat, nlon = memb_data.shape
                    anom = memb_data - memb_data.mean(axis=(1,2), keepdims=True)
                    anom = anom.reshape(nt, -1)
                    ranked = np.argsort(np.argsort(anom, axis=0), axis=0)
                    corr_matrix = np.corrcoef(ranked.T)
                    corr_vs_dist = np.array([
                        np.nanmean(
                            corr_matrix[(dist_matrix >= bins[i]) & (dist_matrix < bins[i+1])]
                        )
                        for i in range(len(bins)-1)
                    ])
                    plt.plot(
                        0.5*(bins[:-1]+bins[1:]),
                        corr_vs_dist,
                        color=color,
                        alpha=0.3,
                    )

            # --- Moyenne ---
            plt.plot(
                0.5*(bins[:-1]+bins[1:]),
                corr_vs_dist_dict[(v, m)],
                color=color,
                label=m
            )

        plt.axhline(0, color='k', lw=0.5)
        plt.grid(alpha=0.3)
        plt.ylim(-0.35, 1.05)

        plt.xlabel('Distance (km)')
        plt.ylabel('Spearman correlation')
        plt.title(f'Spatial Spearman correlation vs distance – members – {v}')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/spatial_spearman_distance_members_{v}.png', dpi=300)
        plt.show()

def era5_threshold(rea, v, q=0.95):
    return rea[f'ERA5 {v}'].quantile(q, dim='time')

def extreme_frequency(da, threshold):
    '''
    da : xr.DataArray (time, lat, lon)
    threshold : xr.DataArray (lat, lon)
    '''
    return (da > threshold).mean(dim='time')

def extreme_intensity(da, threshold):
    masked = da.where(da > threshold)
    return masked.mean(dim='time')

def plot_extreme_frequency_subplots(rea, variables, methods, plot_dir, q=0.95):
    '''
    Subplots per variable of extreme frequency for all methods.
    '''
    nvar = len(variables)
    ncols = 2
    nrows = int(np.ceil(nvar / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    axes = axes.flatten()

    for i, v in enumerate(variables):
        ax = axes[i]

        # ERA5 threshold
        thr = rea[f'ERA5 {v}'].quantile(q, dim='time')

        freqs = []
        labels = ['ERA5'] + methods

        for m in labels:
            f = (rea[f'{m} {v}'] > thr).mean(dim='time')
            freqs.append(f.mean().compute().item())

        ax.bar(labels, freqs)
        ax.set_title(f'{v} – Extreme frequency (q={int(q*100)})')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=45)

    # Remove unused axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Extreme frequency per variable', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/extreme_frequency_subplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_extreme_intensity_subplots(rea, variables, methods, plot_dir, q=0.95):
    '''
    Subplots per variable of extreme intensity for all methods.
    '''
    nvar = len(variables)
    ncols = 2
    nrows = int(np.ceil(nvar / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    axes = axes.flatten()

    for i, v in enumerate(variables):
        ax = axes[i]

        thr = rea[f'ERA5 {v}'].quantile(q, dim='time')

        intensities = []
        labels = ['ERA5'] + methods

        for m in labels:
            da = rea[f'{m} {v}']
            intensity = da.where(da > thr).mean(dim='time')
            intensities.append(intensity.mean().compute().item())

        ax.bar(labels, intensities)
        
        # Ajuster l'axe Y pour mieux voir les différences
        ymin = min(intensities) * 0.95  # 5% en dessous du minimum
        ymax = max(intensities) * 1.05  # 5% au dessus du maximum
        ax.set_ylim(ymin, ymax)
        
        ax.set_title(f'{v} – Extreme intensity (q={int(q*100)})')
        ax.set_ylabel('Intensity')
        ax.tick_params(axis='x', rotation=45)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Extreme intensity per variable', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/extreme_intensity_subplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def global_annual_anomaly(da):
    '''
    da: DataArray (time, lat, lon)
    return: DataArray (year)
    '''
    annual_mean = da.resample(time='YE').mean()
    global_mean = annual_mean.mean(dim=['lat','lon'])
    climatology = global_mean.mean(dim='time')
    return global_mean - climatology

def plot_global_annual_anomalies(gcm, variables, gcm_methods, plot_dir, members):
    '''
    Plot global annual anomalies:
    one subplot per variable, all methods + GCM on each subplot.
    Members are plotted in faint color (alpha=0.2) for each method.
    '''
    if members is not None:
        print(f'MEMBERS VARS: {members.data_vars}')

    years = gcm.time.dt.year.groupby('time.year').mean('time').year

    n_vars = len(variables)
    ncols = 2
    nrows = int(np.ceil(n_vars / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6*ncols, 4*nrows),
        sharex=True
    )
    axes = axes.flatten()

    import matplotlib
    colors = plt.get_cmap('tab10', len(gcm_methods))

    for i, v in enumerate(variables):
        ax = axes[i]

        # --- GCM (référence) ---
        anom_gcm = global_annual_anomaly(gcm[f'GCM {v}'])
        ax.plot(years, anom_gcm, label='GCM', color='black')

        # --- Méthodes ---
        for idx, method in enumerate(gcm_methods):
            anom_model = global_annual_anomaly(gcm[f'{method} {v}'])
            ax.plot(years, anom_model, label=method, color=colors(idx))

            # --- Membres ---
            if members is not None:
                if f'{method} {v}' in members.data_vars:
                    member_vars = members[f'{method} {v}']
                    anom_member = global_annual_anomaly(member_vars)  # shape = (member, time) ou (member, year)
                    for m_idx in range(anom_member.sizes['member']):
                        ax.plot(
                            years,
                            anom_member.isel(member=m_idx),
                            color=colors(idx),
                            alpha=0.3,
                            zorder=1
                        )
        ax.set_title(v)
        ax.set_xlabel('Year')
        ax.set_ylabel(v)
        ax.grid(alpha=0.3)

    # Supprimer axes vides
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j])

    # Légende commune
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(gcm_methods)+1,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'{plot_dir}/global_annual_anomalies_all_vars.png', dpi=300)
    plt.close()

def orographic_variogram_metric(
    rea,
    variables,
    methods,
    lat,
    lon,
    orog,
    alt_min=800.0,
    nbins=40,
    hmax_metric=200.0,
):
    '''
    Compute orographic variograms and a multi-variable spatial metric.

    Returns
    -------
    variogram_dict : dict
        Keys = (variable, method), values = variogram
    spatial_metrics : dict
        One scalar metric per method (mean over variables)
    bins : ndarray
        Distance bin centers
    dist_matrix : ndarray
        Pairwise distance matrix (mountain points only)
    '''

    # ---------- mask montagne ----------
    mask = orog.values > alt_min
    idx = np.where(mask.ravel())[0]

    # ---------- distances ----------
    dist_full = compute_grid_distances(lat, lon)
    dist_matrix = dist_full[np.ix_(idx, idx)]

    # ---------- bins ----------
    iu = np.triu_indices(len(idx), k=1)
    dists = dist_matrix[iu]
    bins = np.linspace(0, dists.max(), nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    variogram_dict = {}
    spatial_metrics = {}

    # ---------- ERA5 reference ----------
    ref_variograms = {}

    for v in variables:
        data = rea[f'ERA5 {v}'].values
        nt = data.shape[0]

        anom = data - data.mean(axis=(1, 2), keepdims=True)
        anom = anom.reshape(nt, -1)[:, idx]
        field = anom.mean(axis=0)

        gamma = 0.5 * (field[iu[0]] - field[iu[1]]) ** 2

        vario = np.array([
            np.nanmean(gamma[(dists >= bins[i]) & (dists < bins[i + 1])])
            for i in range(nbins)
        ])

        ref_variograms[v] = vario
        variogram_dict[(v, 'ERA5')] = vario

    # ---------- methods ----------
    for m in methods:
        scores = []

        for v in variables:
            data = rea[f'{m} {v}'].values
            nt = data.shape[0]

            anom = data - data.mean(axis=(1, 2), keepdims=True)
            anom = anom.reshape(nt, -1)[:, idx]
            field = anom.mean(axis=0)

            gamma = 0.5 * (field[iu[0]] - field[iu[1]]) ** 2

            vario = np.array([
                np.nanmean(gamma[(dists >= bins[i]) & (dists < bins[i + 1])])
                for i in range(nbins)
            ])

            variogram_dict[(v, m)] = vario

            # ----- métrique (normalisée ERA5) -----
            mask_h = bin_centers <= hmax_metric
            score = np.sqrt(
                np.nanmean(
                    (vario[mask_h] - ref_variograms[v][mask_h]) ** 2
                )
            ) / np.nanmean(ref_variograms[v][mask_h])

            scores.append(score)
        spatial_metrics[f"{m} sfcWind"] = scores[0]
        spatial_metrics[m] = np.mean(scores)

    return variogram_dict, spatial_metrics, bin_centers, dist_matrix

def plot_orographic_variograms(
    variogram_dict,
    bins,
    variables,
    methods,
    plot_dir=None,
    title='Orographic variograms (elevation > threshold)'
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    for ax, v in zip(axes, variables):
        ref = variogram_dict[(v, 'ERA5')]
        ref_norm = np.nanmean(ref)

        ax.plot(
            bins,
            ref / ref_norm,
            'k--',
            lw=1,
            label='ERA5'
        )

        for m in methods:
            ax.plot(
                bins,
                variogram_dict[(v, m)] / ref_norm,
                label=m
            )

        ax.set_title(v)
        ax.grid(alpha=0.3)

    axes[2].set_xlabel('Horizontal distance (km)')
    axes[3].set_xlabel('Horizontal distance (km)')
    axes[0].set_ylabel('Normalized semivariogram')
    axes[2].set_ylabel('Normalized semivariogram')

    axes[0].legend(ncol=2, fontsize=9)
    plt.suptitle(title)
    plt.tight_layout()

    if plot_dir is not None:
        plt.savefig(
            f'{plot_dir}/orographic_variograms.png',
            dpi=200,
            bbox_inches='tight'
        )
    plt.close()

def plot_cdf_subplot_final(rea, v, methods, mask=None, name='global', members=None):
    '''
    Plot CDFs en 3 subplots : full, quantiles>0.9, quantiles<0.1.
    La distribution d'une méthode correspond à la distribution de tous ses membres concaténés.
    Évite les doublons si la méthode est déjà dans `rea`.
    '''
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    # --- ERA5 ---
    era5 = rea[f'ERA5 {v}']
    if mask is not None:
        era5 = era5.where(mask)
    era5_vals = np.sort(era5.values.flatten())
    era5_vals = era5_vals[~np.isnan(era5_vals)]
    cdf_era5 = np.linspace(0, 1, len(era5_vals))
    
    axes[0].plot(era5_vals, cdf_era5, label='ERA5', color='k')
    axes[1].plot(era5_vals[cdf_era5>0.95], cdf_era5[cdf_era5>0.95], label='ERA5', color='k')
    axes[2].plot(era5_vals[cdf_era5<0.05], cdf_era5[cdf_era5<0.05], label='ERA5', color='k')

    colors = plt.get_cmap('tab10', len(methods))

    for idx, m in enumerate(methods):
        if '1m' not in m:
            da = rea[f'{m} {v}']
            if mask is not None:
                da = da.where(mask)
            vals = np.sort(da.values.flatten())
            vals = vals[~np.isnan(vals)]
            cdf_vals = np.linspace(0,1,len(vals))

            for ax, qmask in zip(axes, [slice(None), cdf_vals>0.95, cdf_vals<0.05]):
                ax.plot(vals[qmask], cdf_vals[qmask], label=m, color=colors(idx))

            if members is not None:
                if f'{m} {v}' in members.data_vars:
                    vals = members[f'{m} {v}'].values
                    vals = np.sort(da.values.flatten())
                    vals = vals[~np.isnan(vals)]
                    cdf_vals = np.linspace(0,1,len(vals))
                    for ax, qmask in zip(axes, [slice(None), cdf_vals>0.95, cdf_vals<0.05]):
                        ax.plot(vals[qmask], cdf_vals[qmask], color=colors(idx), linestyle='--', label=f'{m} members')
    for ax in axes:
        ax.set_xlabel(v)
        ax.set_ylabel('CDF')
    axes[0].set_title('Full CDF')
    axes[1].set_title('Quantiles > 0.95')
    axes[2].set_title('Quantiles < 0.05')

    # Légende centrée sous les subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=len(methods)*2, frameon=False)

    plt.suptitle(f'CDF {v} – {name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(f'{plot_dir}/full_cdf_{v}_{name}_subplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def orographic_spatial_correlation_metric(
    rea,
    variables,
    methods,
    lat,
    lon,
    orog,
    alt_min=800.0,
    nbins=40,
    h0_metric=25.0,   # km : échelle locale critique
):

    # ---------- mask montagne ----------
    mask = orog.values > alt_min
    idx = np.where(mask.ravel())[0]

    # ---------- distances ----------
    dist_full = compute_grid_distances(lat, lon)
    dist_matrix = dist_full[np.ix_(idx, idx)]

    iu = np.triu_indices(len(idx), k=1)
    dists = dist_matrix[iu]

    bins = np.linspace(0, dists.max(), nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    corr_vs_dist_dict = {}
    spatial_metrics = {}

    # ---------- ERA5 reference ----------
    corr_ref = {}

    for v in variables:
        data = rea[f'ERA5 {v}'].values
        nt = data.shape[0]

        anom = data - data.mean(axis=(1, 2), keepdims=True)
        anom = anom.reshape(nt, -1)[:, idx]

        ranked = np.argsort(np.argsort(anom, axis=0), axis=0)
        corr_matrix = np.corrcoef(ranked.T)

        corr_vs_dist = np.array([
            np.nanmean(corr_matrix[iu][(dists >= bins[i]) & (dists < bins[i+1])])
            for i in range(nbins)
        ])

        corr_ref[v] = corr_matrix
        corr_vs_dist_dict[(v, 'ERA5')] = corr_vs_dist

    # ---------- methods ----------
    for m in methods:
        scores = []

        for v in variables:
            data = rea[f'{m} {v}'].values
            nt = data.shape[0]

            anom = data - data.mean(axis=(1, 2), keepdims=True)
            anom = anom.reshape(nt, -1)[:, idx]

            ranked = np.argsort(np.argsort(anom, axis=0), axis=0)
            corr_matrix = np.corrcoef(ranked.T)

            corr_vs_dist = np.array([
                np.nanmean(corr_matrix[iu][(dists >= bins[i]) & (dists < bins[i+1])])
                for i in range(nbins)
            ])

            corr_vs_dist_dict[(v, m)] = corr_vs_dist

            # ----- local over-correlation metric -----
            mask_h = dists <= h0_metric
            score = np.abs(
                np.nanmean(corr_matrix[iu][mask_h]) -
                np.nanmean(corr_ref[v][iu][mask_h])
            )

            scores.append(score)
        spatial_metrics[f"{m} sfcWind"] = scores[0]
        spatial_metrics[m] = np.mean(scores)

    return corr_vs_dist_dict, spatial_metrics, bin_centers, dist_matrix

def plot_orographic_spatial_correlations(
    corr_vs_dist_dict,
    bins,
    variables,
    methods,
    plot_dir=None,
    title='Spatial correlation over mountainous regions'
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    for ax, v in zip(axes, variables):

        ax.plot(
            bins,
            corr_vs_dist_dict[(v, 'ERA5')],
            'k--',
            lw=1,
            label='ERA5'
        )

        for m in methods:
            ax.plot(
                bins,
                corr_vs_dist_dict[(v, m)],
                label=m
            )

        ax.axvline(25, color='gray', linestyle=':', alpha=0.6)
        ax.set_title(v)
        ax.grid(alpha=0.3)

    axes[2].set_xlabel('Horizontal distance (km)')
    axes[3].set_xlabel('Horizontal distance (km)')
    axes[0].set_ylabel('Mean spatial correlation')
    axes[2].set_ylabel('Mean spatial correlation')

    axes[0].legend(ncol=2, fontsize=9)
    plt.suptitle(title)
    plt.tight_layout()

    if plot_dir is not None:
        plt.savefig(
            f'{plot_dir}/orographic_spatial_correlation.png',
            dpi=200,
            bbox_inches='tight'
        )
    plt.show()

def isotropic_spatial_spectrum(field2d, dx, dy):
    '''
    Compute isotropic 2D spatial power spectrum of a 2D field.
    '''
    ny, nx = field2d.shape

    fft2 = np.fft.fft2(field2d)
    psd2 = np.abs(fft2) ** 2

    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kx2d, ky2d = np.meshgrid(kx, ky)
    k = np.sqrt(kx2d**2 + ky2d**2)

    return k.ravel(), psd2.ravel()

def spatial_spectrum_and_metric(
    rea,
    variables,
    methods,
    lat,
    lon,
    nbins=40,
    kmax_metric=None,
):
    '''
    Compute spatial power spectra and a spectral coherence metric.

    Returns
    -------
    spectrum_dict : dict
        (variable, method) -> mean spatial spectrum
    spectral_metrics : dict
        One scalar metric per method (mean over variables)
    k_centers : ndarray
        Wavenumber bin centers
    '''

    dx = np.mean(np.diff(lon)) * 111e3
    dy = np.mean(np.diff(lat)) * 111e3

    k_bins = np.logspace(-6, -4, nbins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    spectrum_dict = {}
    spectral_metrics = {}

    # ---------- ERA5 reference ----------
    ref_spectra = {}

    for v in variables:
        data = rea[f'ERA5 {v}'].values
        nt = data.shape[0]

        spectra = []

        for t in range(nt):
            field = data[t]
            field = field - field.mean()
            field = np.nan_to_num(field)

            k, psd = isotropic_spatial_spectrum(field, dx, dy)

            spec = np.array([
                np.nanmean(psd[(k >= k_bins[i]) & (k < k_bins[i+1])])
                for i in range(nbins)
            ])
            spectra.append(spec)

        ref_spectra[v] = np.nanmean(spectra, axis=0)
        spectrum_dict[(v, 'ERA5')] = ref_spectra[v]

    # ---------- methods ----------
    for m in methods:
        scores = []

        for v in variables:
            data = rea[f'{m} {v}'].values
            nt = data.shape[0]

            spectra = []

            for t in range(nt):
                field = data[t]
                field = field - field.mean()
                field = np.nan_to_num(field)

                k, psd = isotropic_spatial_spectrum(field, dx, dy)

                spec = np.array([
                    np.nanmean(psd[(k >= k_bins[i]) & (k < k_bins[i+1])])
                    for i in range(nbins)
                ])
                spectra.append(spec)

            spec = np.nanmean(spectra, axis=0)
            spectrum_dict[(v, m)] = spec

            # ----- metric: large-scale excess energy -----
            if kmax_metric is None:
                kmax_metric = k_centers[int(nbins * 0.3)]

            mask_k = k_centers <= kmax_metric

            score = np.sqrt(
                np.nanmean((spec[mask_k] - ref_spectra[v][mask_k])**2)
            ) / np.nanmean(ref_spectra[v][mask_k])

            scores.append(score)
        spectral_metrics[f"{m} sfcWind"] = scores[0]
        spectral_metrics[m] = np.mean(scores)

    return spectrum_dict, spectral_metrics, k_centers

def plot_spatial_spectra_subplot(
    spectrum_dict,
    k_centers,
    variables,
    methods,
    plot_dir='../data/plots'
):
    colors = plt.get_cmap('tab10', len(methods))
    nvars = len(variables)
    ncols = 2
    nrows = int(np.ceil(nvars / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7*ncols, 5*nrows),
        sharex=True, sharey=True
    )
    axes = axes.ravel()

    for ax, v in zip(axes, variables):

        # --- ERA5 ---
        ax.loglog(
            k_centers,
            spectrum_dict[(v, 'ERA5')],
            'k',
            lw=1,
            label='ERA5'
        )

        # --- Methods ---
        for i, m in enumerate(methods):
            ax.loglog(
                k_centers,
                spectrum_dict[(v, m)],
                color=colors(i),
                label=m
            )

        ax.set_title(v)
        ax.grid(which='both', alpha=0.3)

        # ---------- top axis: spatial scale ----------
        ax_top = ax.twiny()
        ax_top.set_xscale('log')

        kmin, kmax = ax.get_xlim()
        ax_top.set_xlim(kmin, kmax)

        # ticks en km
        scale_ticks_km = np.array([20, 50, 100, 200, 500])
        k_ticks = 1.0 / (scale_ticks_km * 1000.0)

        ax_top.set_xticks(k_ticks)
        ax_top.set_xticklabels(scale_ticks_km.astype(int))
        ax_top.set_xlabel('Spatial scale (km)')

    # supprimer axes vides
    for j in range(len(variables), len(axes)):
        fig.delaxes(axes[j])

    fig.text(0.5, 0.04, 'Spatial wavenumber (m$^{-1}$)', ha='center')
    fig.text(0.04, 0.5, 'Power spectral density', va='center', rotation='vertical')

    fig.suptitle('Spatial power spectra – all variables', fontsize=16)

    fig.subplots_adjust(
        bottom=0.15, top=0.92,
        left=0.12, right=0.95,
        hspace=0.3, wspace=0.3
    )

    fig.legend(
        labels=['ERA5'] + methods,
        loc='lower center',
        ncol=len(methods) + 1,
        frameon=False,
        fontsize=10
    )

    plt.savefig(
        f'{plot_dir}/spatial_spectra_all_variables.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

def local_variogram_radius_metric(
    rea,
    variables,
    methods,
    lat,
    lon,
    radius_km=50.0,
    orog=None,
    alt_min=None,
):
    '''
    Compute local spatial variability (variogram-like) at fixed radius.

    Returns
    -------
    delta_vario_maps : dict
        (variable, method) -> 2D map of variability difference vs ERA5
    spatial_metrics : dict
        One scalar metric per method (mean over variables)
    '''

    # ---------- distances ----------
    dist = compute_grid_distances(lat, lon)  # km
    npoints = dist.shape[0]

    # neighbors mask
    neigh = dist <= radius_km
    np.fill_diagonal(neigh, False)

    delta_vario_maps = {}
    spatial_metrics = {}

    # ---------- ERA5 reference ----------
    ref_maps = {}

    for v in variables:
        data = rea[f'ERA5 {v}'].values  # (t, lat, lon)
        nt, nlat, nlon = data.shape
        data = data.reshape(nt, -1)

        ref_local = np.full(npoints, np.nan)

        for i in range(npoints):
            idx = neigh[i]
            if idx.sum() < 5:
                continue
            diffs = data[:, idx] - data[:, i][:, None]
            ref_local[i] = np.nanmean(np.var(diffs, axis=0))

        ref_maps[v] = ref_local.reshape(nlat, nlon)
        delta_vario_maps[(v, 'ERA5')] = ref_maps[v]

    # ---------- methods ----------
    for m in methods:
        scores = []

        for v in variables:
            data = rea[f'{m} {v}'].values
            nt, nlat, nlon = data.shape
            data = data.reshape(nt, -1)

            local_var = np.full(npoints, np.nan)

            for i in range(npoints):
                idx = neigh[i]
                if idx.sum() < 5:
                    continue
                diffs = data[:, idx] - data[:, i][:, None]
                local_var[i] = np.nanmean(np.var(diffs, axis=0))

            local_var = local_var.reshape(nlat, nlon)
            delta = np.abs(local_var - ref_maps[v])
            delta_vario_maps[(v, m)] = delta

            # ----- metric -----
            if orog is not None and alt_min is not None:
                mask = orog.values > alt_min
                score = np.nanmean(delta[mask])
            else:
                score = np.nanmean(delta)


            scores.append(score)
        spatial_metrics[f"{m} sfcWind"] = scores[0]
        spatial_metrics[m] = np.mean(scores)

    return delta_vario_maps, spatial_metrics

def plot_local_variogram_maps(
    delta_vario_maps,
    variables,
    methods,
    lon,
    lat,
    plot_dir,
    shapefile_path=shapefile_path,
    vmax=None,
    colorbar_position='top'  # 'right' ou 'top'
):
    import matplotlib.gridspec as gridspec

    for v in variables:

        # ---------- vmax ----------
        if vmax is None:
            all_vals = []
            for m in methods:
                all_vals.append(np.abs(delta_vario_maps[(v, m)]).ravel())
            vmax_loc = np.nanpercentile(np.concatenate(all_vals), 95)
        else:
            vmax_loc = vmax

        nmethods = len(methods)
        fig = plt.figure(figsize=(5 * nmethods, 5))

        if colorbar_position == 'right':
            gs = gridspec.GridSpec(1, nmethods + 1, width_ratios=[1]*nmethods + [0.05], wspace=0.1)
            axes = [fig.add_subplot(gs[i], projection=ccrs.PlateCarree()) for i in range(nmethods)]
            cbar_ax = fig.add_subplot(gs[-1])
        elif colorbar_position == 'top':
            gs = gridspec.GridSpec(2, nmethods, height_ratios=[0.05, 1], hspace=0.25)
            axes = [fig.add_subplot(gs[1, i], projection=ccrs.PlateCarree()) for i in range(nmethods)]
            cbar_ax = fig.add_subplot(gs[0, :])
        else:
            raise ValueError('colorbar_position must be right or top')

        for ax, m in zip(axes, methods):
            im = ax.pcolormesh(
                lon, lat, delta_vario_maps[(v, m)],
                cmap='BrBG',  # mieux pour valeurs absolues
                vmin=0,
                vmax=vmax_loc,
                transform=ccrs.PlateCarree()
            )

            if shapefile_path is not None:
                coastline = cfeature.ShapelyFeature(
                    Reader(shapefile_path).geometries(),
                    ccrs.PlateCarree(),
                    facecolor='none',
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_feature(coastline)
            else:
                ax.coastlines(linewidth=0.5)

            # Grille
            ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

            ax.set_title(f'{m} - {v}', fontsize=12)
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

        # Colorbar
        if colorbar_position == 'right':
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        else:
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

        cbar.set_label('Absolute local variability difference vs ERA5', fontsize=12)
        fig.suptitle(f'Local spatial variability (radius ≈ 50 km) – {v}',
                     fontsize=14, fontweight='bold', y=0.97)

        plt.savefig(f'{plot_dir}/local_variogram_cartopy_{colorbar_position}_{v}.png',
                    dpi=200, bbox_inches='tight')
        plt.close()
        print(f'Saved: local_variogram_cartopy_{colorbar_position}_{v}.png')

def plot_wind_maps_single_time(
    rea,
    variables,
    methods,
    lon,
    lat,
    time_index=0,
    plot_dir='../data/plots',
    shapefile_path=None,
    cmap='cividis'
):

    all_methods = ['ERA5'] + methods
    nrows = len(variables)
    ncols = len(all_methods)

    # ---- figure & gridspec (2 rows per variable: plot + colorbar)
    fig = plt.figure(figsize=(4.5 * ncols, 4.2 * nrows))

    gs = gridspec.GridSpec(
        nrows * 2,
        ncols,
        height_ratios=[1, 0.06] * nrows,
        hspace=0.25,
        wspace=0.05
    )

    axes = np.empty((nrows, ncols), dtype=object)
    cbar_axes = []

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(
                gs[2 * i, j],
                projection=ccrs.PlateCarree()
            )
        cbar_axes.append(fig.add_subplot(gs[2 * i + 1, :]))

    # ---- plotting
    for i, v in enumerate(variables):

        # ---------- common color scale per variable ----------
        all_vals = []
        for m in all_methods:
            all_vals.append(
                rea[f'{m} {v}'].isel(time=time_index).values.ravel()
            )

        vmin = np.nanpercentile(np.concatenate(all_vals), 5)
        vmax = np.nanpercentile(np.concatenate(all_vals), 95)

        for j, m in enumerate(all_methods):
            ax = axes[i, j]

            data = rea[f'{m} {v}'].isel(time=time_index)

            im = ax.pcolormesh(
                lon, lat, data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree()
            )

            # ---------- borders / coastlines ----------
            if shapefile_path is not None:
                borders = cfeature.ShapelyFeature(
                    Reader(shapefile_path).geometries(),
                    ccrs.PlateCarree(),
                    facecolor='none',
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_feature(borders)
            else:
                ax.coastlines(linewidth=0.5)

            ax.gridlines(
                draw_labels=False,
                linewidth=0.3,
                alpha=0.4,
                linestyle='--'
            )

            ax.set_extent(
                [lon.min(), lon.max(), lat.min(), lat.max()],
                crs=ccrs.PlateCarree()
            )

            # ---------- titles / labels ----------
            if i == 0:
                ax.set_title(m, fontsize=12, fontweight='bold')

            if j == 0:
                ax.text(
                    -0.08, 0.5, v,
                    va='center',
                    ha='right',
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight='bold'
                )

        # ---------- colorbar (dedicated axis, no overlap) ----------
        cbar = fig.colorbar(
            im,
            cax=cbar_axes[i],
            orientation='horizontal'
        )
        cbar.set_label(v)

    # ---- main title
    fig.suptitle(
        f'Wind fields at time index {time_index}',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    # ---- save
    plt.savefig(
        f'{plot_dir}/wind_maps_time{time_index}.png',
        dpi=200,
        bbox_inches='tight'
    )
    plt.close()

def plot_members_methods_same_window(
    gcm,
    members,
    lat,
    lon,
    methods,
    start_date,
    vname,
    window=20,
    plot_dir="../data/plots"
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    start_date = np.datetime64(start_date)
    end_date = start_date + np.timedelta64(window, "D")

    # --- Extraction du point spatial ---
    gcm_pt = gcm.sel(lat=lat, lon=lon, method="nearest").sel(time=slice(start_date, end_date))
    members_pt = members.sel(lat=lat, lon=lon, method="nearest").sel(time=slice(start_date, end_date))

    colors = plt.get_cmap("tab10")

    for i, (ax, method) in enumerate(zip(axes, methods)):

        color = colors(i)
        gcm_slice = gcm_pt[f'GCM {vname}']
        ax.plot(
            gcm_slice.time,
            gcm_slice,
            color="k",
            lw=1,
            label="GCM" if i == 0 else None
        )

        # --- Moyenne des membres (depuis GCM, variable correspondant à la méthode) ---
        mean_slice = gcm_pt[f"{method} {vname}"]
        ax.plot(
            mean_slice.time,
            mean_slice,
            color=color,
            lw=1,
            label=f"{method}"
        )

        for i in range(10):
            mem_slice = members_pt[f"{method} {vname}"].sel(member=i)
            if "Dual" in method:
                v0 = mem_slice.isel(time=0).values
                v1 = mem_slice.isel(time=1).values
                print(f"{method} {i}: {v0}, {v1}")

            ax.plot(
                mem_slice.time,
                mem_slice,
                color=color,
                alpha=0.25,
                lw=1
            )

        ax.set_title(method)
        ax.grid(alpha=0.3)

        if i == 0:
            ax.legend()

    fig.suptitle(
        f"{vname} – {pd.to_datetime(str(start_date)).date()} → "
        f"{pd.to_datetime(str(end_date)).date()} – "
        f"lat={lat:.2f}, lon={lon:.2f}",
        fontsize=15
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(f"{plot_dir}/ts_{vname}.png")
    plt.close()


rea = xr.open_zarr(out_mean, consolidated=True).sel(time=slice('1999-01-01', '2009-01-24'))
compute_nans(rea)
members = xr.open_zarr(out_members, consolidated=True).sel(time=slice('1999-01-01', '2009-01-24'))
compute_nans(members)
methods.append(f'SF {distance[1]} 1m')
for v in variables:
    rea[f'SF {distance[1]} 1m {v}'] = members[f'SF {distance[1]} {v}'].sel(member=0)
methods.append(f'Dual FM 1m')
for v in variables:
    rea[f'Dual FM 1m {v}'] = members[f'Dual FM {v}'].sel(member=0)
methods.append(f"Dual SDE 1m")
for v in variables:
    rea[f'Dual SDE 1m {v}'] = members[f'Dual SDE {v}'].sel(member=0)
methods.append(f"SI 1m")
for v in variables:
    rea[f'SI 1m {v}'] = members[f'SI {v}'].sel(member=0)



if 'GCM sfcWind' not in rea.data_vars:
    for v in variables:
        rea[f'GCM {v}'] = rea[v]
        rea = rea.drop_vars(v)

orog = xr.open_dataset('data/orog_ERA5.nc')['z']
g = 9.80665
orog = orog / g
# --------------------------
# 1. Metrics vs ERA5
# --------------------------
print('Computing metrics vs ERA5...')
plot_wind_maps_single_time(
    rea=rea,
    variables=variables,
    methods=methods,
    lon=rea.lon.values,
    lat=rea.lat.values,
    time_index=0,
    plot_dir=plot_dir,
    shapefile_path=shapefile_path
)
metrics_vs_era5 = {}

print('Mean and std differences...')
metrics_vs_era5 = plot_mean_std_diffs(rea, variables, methods, plot_dir)
for method in methods:
    diffs = [compute_global_mean_abs_diff(rea[f'{method} {v}'], rea[f'ERA5 {v}']) for v in variables]
    metrics_vs_era5[f'{method}_mean'] = np.mean(diffs)
    diffs_std = [compute_global_std_diff(rea[f'{method} {v}'], rea[f'ERA5 {v}']) for v in variables]
    metrics_vs_era5[f'{method}_std'] = np.mean(diffs_std)
print(f'Mean and std differences done in {time.time()-now} seconds.')
now = time.time()
print('Spatial spectrum')

spectrum_dict, spectral_metrics, k_centers = spatial_spectrum_and_metric(
    rea,
    variables=variables,
    methods=methods,
    lat=rea.lat.values,
    lon=rea.lon.values,
    nbins=40
)
for m in methods:
    metrics_vs_era5[f'{m}_spatial_spectrum'] = spectral_metrics[m]
    metrics_vs_era5[f'{m}_spatial_spectrum_sfcWind'] = spectral_metrics[f"{m} sfcWind"]

plot_spatial_spectra_subplot(spectrum_dict, k_centers, variables, methods, plot_dir=plot_dir)

print(f'Spatial spectrum tool {time.time()-now:.1f} seconds')
now = time.time()
print('Variogram maps')
delta_maps, spatial_metrics = local_variogram_radius_metric(
    rea,
    variables=variables,
    methods=methods,
    lat=rea.lat.values,
    lon=rea.lon.values,
    radius_km=50,
    orog=None,
    alt_min=None
)

for m in methods:
    metrics_vs_era5[f'{m}_local_vario50'] = spatial_metrics[m]
    metrics_vs_era5[f'{m}_local_vario50_sfcWind'] = spatial_metrics[f"{m} sfcWind"]

plot_local_variogram_maps(
    delta_maps,
    variables=variables,
    methods=methods,
    lat=rea.lat.values,
    lon=rea.lon.values,
    plot_dir=plot_dir,
    vmax=4
)

delta_maps, spatial_metrics = local_variogram_radius_metric(
    rea,
    variables=variables,
    methods=methods,
    lat=rea.lat.values,
    lon=rea.lon.values,
    radius_km=50,
    orog=orog,
    alt_min=800
)
for m in methods:
    metrics_vs_era5[f'{m}_local_vario50_800'] = spatial_metrics[m]
    metrics_vs_era5[f'{m}_local_vario50_800_sfcWind'] = spatial_metrics[f"{m} sfcWind"]
print(f'Variogram maps done in {time.time()-now:.1f} seconds')

now = time.time()
print(f'Variogram')

variogram_dict, spatial_metrics, bins, dist_matrix = (
    orographic_variogram_metric(
        rea,
        variables=variables,
        methods=methods,
        lat=rea.lat.values,
        lon=rea.lon.values,
        orog=orog,
        alt_min=300,
        nbins=40
    )
)
for m in methods:
    metrics_vs_era5[f'{m}_variogram'] = spatial_metrics[m]
    metrics_vs_era5[f'{m}_variogram_sfcWind'] = spatial_metrics[f"{m} sfcWind"]
plot_orographic_variograms(
    variogram_dict,
    bins=bins,
    variables=variables,
    methods=methods,
    plot_dir=plot_dir
)
print(f'Variogram tool {time.time()-now} seconds')
now = time.time()
print('Spatial correlation (mountain)')

corr_vs_dist_dict, spatial_metrics, bins, dist_matrix = (
    orographic_spatial_correlation_metric(
        rea,
        variables=variables,
        methods=methods,
        lat=rea.lat.values,
        lon=rea.lon.values,
        orog=orog,
        alt_min=800,
        nbins=40,
        h0_metric=25
    )
)

for m in methods:
    metrics_vs_era5[f'{m}_local_overcorr'] = spatial_metrics[m]
    metrics_vs_era5[f'{m}_local_overcorr_sfcWind'] = spatial_metrics[f"{m} sfcWind"]

plot_orographic_spatial_correlations(
    corr_vs_dist_dict,
    bins=bins,
    variables=variables,
    methods=methods,
    plot_dir=plot_dir
)

print(f'Spatial correlation tool {time.time()-now:.1f} seconds')

now = time.time()
print('Inter variable correlations...')
plot_intervar_corr_grid(rea, variables, methods, plot_dir, corr_type='pearson')
# Correlation differences (Pearson)
var_pairs = list(itertools.combinations(variables, 2))
for method in methods:
    corrs_diff = []
    for v1,v2 in var_pairs:
        corr_model = xr.corr(rea[f'{method} {v1}'], rea[f'{method} {v2}'], dim='time')
        corr_era5 = xr.corr(rea[f'ERA5 {v1}'], rea[f'ERA5 {v2}'], dim='time')
        corrs_diff.append(np.abs(corr_model - corr_era5).mean().compute().item())
    metrics_vs_era5[f'{method}_corr_pairs'] = np.mean(corrs_diff)
print(f'Inter variable correlations done in {time.time()-now} seconds.')
now = time.time()

print('Spatial Spearman correlation')
print('Spatial Spearman correlation (vectorized)...')
corr_vs_dist_dict, spatial_metrics, bins, dist_matrix = spatial_spearman_distance_and_metric(
    rea, variables, methods, rea.lat.values, rea.lon.values, nbins=40
)

plot_spatial_spearman_means(corr_vs_dist_dict, bins=bins, variables=variables, methods=methods, plot_dir=plot_dir)
plot_spatial_spearman_members(corr_vs_dist_dict, dist_matrix, bins, variables, 
                              generative_methods=generative_methods, members=members, plot_dir=plot_dir)

metrics_vs_era5.update(spatial_metrics)
print(f'Spatial correlation done in {time.time()-now} seconds.')
now = time.time()
print('KS statistics...')
# KS statistics
for method in methods:
    ks_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs=rea[f'{method} {v}'].values, method=rea[f'ERA5 {v}'].values) for v in variables)
    metrics_vs_era5[f'{method}_KS'] = np.mean(ks_vals)
    metrics_vs_era5[f'{method}_KS_sfcWind'] = ks_vals[0]

alps = ((rea.lat>44)&(rea.lat<47)&(rea.lon>5)&(rea.lon<8))
med  = ((rea.lat>41)&(rea.lat<43)&(rea.lon>3)&(rea.lon<6))
members_alps = members.where(((members.lat>44)&(members.lat<47)&(members.lon>5)&(members.lon<8)))
members_med = members.where(((members.lat>41)&(members.lat<43)&(members.lon>3)&(members.lon<6)))
for v in variables:
    plot_cdf_subplot(rea, v, methods, alps, 'Alps', members=members_alps)
    plot_cdf_subplot(rea, v, methods, med, 'Mediterranean', members=members_med)
    plot_cdf_subplot(rea, v, methods, None, 'Global', members=members)
members_alps = None
members_med = None
print(f'KS statistics done in {time.time()-now} seconds.')
now = time.time()
print('KS members statistics...')

# KS sur quantiles (queues)
for method in methods:
    if ('Dual' in method) or ('SF' in method):
        if '1m' not in method:
            ks_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, members=members[f'{method} {v}'].values) for v in variables)
            ks_05_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, members=members[f'{method} {v}'].values, quantile = 0.05) for v in variables)
            ks_95_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, members=members[f'{method} {v}'].values, quantile = 0.95) for v in variables)
            for i, v in enumerate(variables):
                metrics_vs_era5[f'{method}_members_KS_full_{v}'] = ks_vals[i]
                metrics_vs_era5[f'{method}_members_KS_q005_{v}'] = ks_05_vals[i]
                metrics_vs_era5[f'{method}_members_KS_q095_{v}'] = ks_95_vals[i]
            ks_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, method=rea[f'{method} {v}'].values) for v in variables)
            ks_05_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, method=rea[f'{method} {v}'].values, quantile = 0.05) for v in variables)
            ks_95_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, method=rea[f'{method} {v}'].values, quantile = 0.95) for v in variables)
            for i, v in enumerate(variables):
                metrics_vs_era5[f'{method}_mean_KS_full_{v}'] = ks_vals[i]
                metrics_vs_era5[f'{method}_mean_KS_q005_{v}'] = ks_05_vals[i]
                metrics_vs_era5[f'{method}_mean_KS_q095_{v}'] = ks_95_vals[i]
        else:
            ks_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, method=rea[f'{method} {v}'].values) for v in variables)
            ks_05_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, method=rea[f'{method} {v}'].values, quantile = 0.05) for v in variables)
            ks_95_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, method=rea[f'{method} {v}'].values, quantile = 0.95) for v in variables)
            for i, v in enumerate(variables):
                metrics_vs_era5[f'{method}_1m_KS_full_{v}'] = ks_vals[i]
                metrics_vs_era5[f'{method}_1m_KS_q005_{v}'] = ks_05_vals[i]
                metrics_vs_era5[f'{method}_1m_KS_q095_{v}'] = ks_95_vals[i]
    else:
        ks_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, method=rea[f'{method} {v}'].values) for v in variables)
        ks_05_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, method=rea[f'{method} {v}'].values, quantile = 0.05) for v in variables)
        ks_95_vals = Parallel(n_jobs=-1)(delayed(compute_ks_stat)(obs = rea[f'ERA5 {v}'].values, method=rea[f'{method} {v}'].values, quantile = 0.95) for v in variables)
        for i, v in enumerate(variables):
            metrics_vs_era5[f'{method}_KS_full_{v}'] = ks_vals[i]
            metrics_vs_era5[f'{method}_KS_q005_{v}'] = ks_05_vals[i]
            metrics_vs_era5[f'{method}_KS_q095_{v}'] = ks_95_vals[i]
for v in variables:
    plot_cdf_subplot_final(rea, v, methods, mask=None, name='Global', members=members)

print(f'KS members statistics done in {time.time()-now} seconds.')
now = time.time()

print('Extremes...')
# Extremes
for method in methods:
    vals = []
    for v in variables:
        thr = era5_threshold(rea, v)
        f_mod = extreme_frequency(rea[f'{method} {v}'], thr)
        f_era = extreme_frequency(rea[f'ERA5 {v}'], thr)
        vals.append(np.abs(f_mod - f_era).mean().compute().item())

    metrics_vs_era5[f'{method}_extreme_freq'] = np.mean(vals)
    metrics_vs_era5[f'{method}_extreme_freq_sfcWind'] = vals[0]

for method in methods:
    vals = []
    for v in variables:
        thr = era5_threshold(rea, v)
        i_mod = extreme_intensity(rea[f'{method} {v}'], thr)
        i_era = extreme_intensity(rea[f'ERA5 {v}'], thr)
        vals.append(np.abs(i_mod - i_era).mean().compute().item())

    metrics_vs_era5[f'{method}_extreme_intensity'] = np.mean(vals)
    metrics_vs_era5[f'{method}_extreme_intensity_sfcWind'] = vals[0]

plot_extreme_frequency_subplots(rea, variables, methods, plot_dir)
plot_extreme_intensity_subplots(rea, variables, methods, plot_dir)
print(f'Extremes done in {time.time()-now} seconds.')
now = time.time()
print('Seasonal cycle...')
# Seasonal cycle
for method in methods:
    seasonal_diffs = []
    for v in variables:
        era5_cycle = rea[f'ERA5 {v}'].groupby('time.month').mean('time')
        model_cycle = rea[f'{method} {v}'].groupby('time.month').mean('time')
        months = era5_cycle.month
        seasonal_diffs.append(np.abs(model_cycle - era5_cycle).mean().compute().item())
    metrics_vs_era5[f'{method}_seasonal'] = np.mean(seasonal_diffs)
    metrics_vs_era5[f'{method}_seasonal_sfcWind'] = seasonal_diffs[0]

plot_seasonal_cycle(rea, variables, methods, plot_dir, members=members)
print(f'Seasonal cycle done in {time.time()-now} seconds.')
now = time.time()

# --------------------------
# 2. Metrics vs GCM
# --------------------------
rea = None
members=None

gcm = xr.open_zarr(out_mean, consolidated=True).sel(time=slice('1999-01-01', '2100-12-31'))
compute_nans(gcm)
members = xr.open_zarr(out_members, consolidated=True).sel(time=slice('1999-01-01', '2100-12-31'))
compute_nans(members)
for v in variables:
    gcm[f'SF {distance[1]} 1m {v}'] = members[f'SF {distance[1]} {v}'].sel(member=0)
    gcm[f'Dual FM 1m {v}'] = members[f'Dual FM {v}'].sel(member=0)
    gcm[f'Dual SDE 1m {v}'] = members[f'Dual SDE {v}'].sel(member=0)
    gcm[f'SI 1m {v}'] = members[f'SI {v}'].sel(member=0)
for v in variables:
    plot_members_methods_same_window(gcm, members, lat=45, lon=2, methods=generative_methods, 
                                     start_date='2040-02-07', vname=v)
print('Computing metrics vs GCM...')
metrics_vs_gcm = {}
gcm_methods = methods[1:]
print('Temporal correlation...')


# Temporal correlation
for method in gcm_methods:
    corr_vals = [xr.corr(gcm[f'{method} {v}'], gcm[f'GCM {v}'], dim='time').mean().compute().item() for v in variables]
    metrics_vs_gcm[f'{method}_corr'] = np.mean(corr_vals)
    metrics_vs_gcm[f'{method}_corr_sfcWind'] = corr_vals[0]

plot_temporal_correlation_subplot(gcm, variables, gcm_methods, plot_dir)
print(f'Temporal correlation done in {time.time()-now} seconds.')
now = time.time()
print('Global annual anomalies...')
# Temporal anomalies
for method in gcm_methods:
    diffs = []
    for v in variables:
        anom_model = global_annual_anomaly(gcm[f'{method} {v}'])
        anom_gcm   = global_annual_anomaly(gcm[f'GCM {v}'])
        diffs.append(np.abs(anom_model - anom_gcm).mean().compute().item())

    metrics_vs_gcm[f'{method}_anom'] = np.mean(diffs)
    metrics_vs_gcm[f'{method}_anom_sfcWind'] = diffs[0]
plot_global_annual_anomalies(gcm, variables, gcm_methods, plot_dir, members=members)
print(f'Global annual anomalies done in {time.time()-now} seconds.')

now = time.time()
print('Delta metrics...')
delta_v = [v for v in variables if 'sfcWind' in v]
for method in gcm_methods:
    deltas_full = []
    deltas_season = []
    for idx,fut_period in enumerate(fut_periods):
        fut_name = fut_names[idx]
        for v in delta_v:
            hist_data = gcm[f'{method} {v}'].sel(time=hist_period)
            fut_data = gcm[f'{method} {v}'].sel(time=fut_period)
            gcm_hist = gcm[f'GCM {v}'].sel(time=hist_period)
            gcm_fut = gcm[f'GCM {v}'].sel(time=fut_period)
            val, delta, gcm_delta = compute_delta_metric(hist_data, fut_data, gcm_hist, gcm_fut)
            deltas_full.append(val)
            # Seasonal
            for season, months in seasons.items():
                val_s, delta_s, gcm_delta_s = compute_delta_metric(hist_data, fut_data, gcm_hist, gcm_fut, months)
                deltas_season.append(val_s)

    metrics_vs_gcm[f'{method}_delta_full'] = np.mean(deltas_full)
    metrics_vs_gcm[f'{method}_delta_season'] = np.mean(deltas_season)
    metrics_vs_gcm[f'{method}_delta_full_sfcWind'] = deltas_full[0]
    metrics_vs_gcm[f'{method}_delta_season_sfcWind'] = deltas_season[0]


plot_delta_variable_grid(gcm, delta_v, gcm_methods, fut_periods, fut_names, hist_period, seasons, plot_dir, seasonal=False)
plot_delta_variable_grid(gcm, delta_v, gcm_methods, fut_periods, fut_names, hist_period, seasons, plot_dir, seasonal=True)
print(f'Delta metrics done in {time.time()-now} seconds.')
now = time.time()
# --------------------------
# Radar plots
# --------------------------
print('Generating radar plots...')
METRIC_DIRECTIONS = {
    'mean': 'min',
    'std': 'min',
    'extreme_freq': 'min',
    'extreme_intensity': 'min',
    'KS': 'min',
    'brier': 'min',
    'seasonal': 'min',
    'anom': 'min',
    'delta_full': 'min',
    'delta_season': 'min',
    'corr_pairs': 'min',
    'spatial_corr': 'min',
    'corr': 'max',
    'variogram': 'min'
}

print(f'\nERA5 Metrics: {metrics_vs_era5}\n')
print(f'GCM Metrics: {metrics_vs_gcm}')
print('\nRadar plots done in', time.time()-now, 'seconds.')
print('All metrics calculated with parallelization and all plots saved in:', plot_dir)