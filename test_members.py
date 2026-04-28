import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def plot_cdf_subplot(members, v, methods, name="Global"):
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    # --- Préparer les méthodes ---
    colors = plt.get_cmap('tab10', len(methods))
    for idx, m in enumerate(methods):
        member_vars = members[f"{m} {v}"]
        for i in range(10):
            mda = member_vars.sel(member=i)
            mvals = np.sort(mda.values.flatten())
            mvals = mvals[~np.isnan(mvals)]
            print(f"{m} {v} {i} - shape: {mda.values.shape} - valeurs min/max: {np.nanmin(mvals):.3f}/{np.nanmax(mvals):.3f}, mean/std: {np.nanmean(mvals):.3f}/{np.std(mvals):.3f}")

            mcdf = np.linspace(0,1,len(mvals))
            for ax, qmask in zip(axes, [slice(None), mcdf>0.9, mcdf<0.1]):
                ax.plot(mvals[qmask], mcdf[qmask], color=colors(idx), alpha=0.3, label=f"{m} {v} {i}")
        mda = member_vars.values
        mvals = np.sort(mda.flatten())
        mvals = mvals[~np.isnan(mvals)]
        mcdf = np.linspace(0,1,len(mvals))
        print(f"{m} {v} {i} - shape: {mda.shape} - valeurs min/max: {np.nanmin(mvals):.3f}/{np.nanmax(mvals):.3f}, mean/std: {np.nanmean(mvals):.3f}/{np.std(mvals):.3f}")

        for ax, qmask in zip(axes, [slice(None), mcdf>0.9, mcdf<0.1]):
            ax.plot(mvals[qmask], mcdf[qmask], color=colors(idx), alpha=1, label=f"{m} {v}")

    # --- Labels ---
    for ax in axes:
        ax.set_xlabel(v)
        ax.set_ylabel("CDF")
    axes[0].set_title(f"Full CDF {name}")
    axes[1].set_title("Quantiles > 0.9")
    axes[2].set_title("Quantiles < 0.1")
    
    # Légende centrée sous les subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=len(methods)+1, frameon=False)

    plt.suptitle(f"CDF {v}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Faire de la place pour la légende
    plt.savefig(f"{plot_dir}/cdf_{name}_{v}_test_members_subplots.png", dpi=300, bbox_inches='tight')
    plt.close()

print(f"Load members")
data_path = "data"
plot_dir = "../data/plots"
out_members = f"{data_path}/results/final_ds_members.zarr"
members = xr.open_zarr(out_members, consolidated=True).sel(time=slice("1999-01-01", "2009-01-24"))
"""for v in ['sfcWind', 'sfcWindmax', 'uas', 'vas']:
    print(v)
    plot_cdf_subplot(members, v, ['Dual FM', 'SF 1200 km', 'SF 750 km', 'SF 300 km'])
"""
members_alps = members.where(((members.lat>44)&(members.lat<47)&(members.lon>5)&(members.lon<8)))
members_med = members.where(((members.lat>41)&(members.lat<43)&(members.lon>3)&(members.lon<6)))
for v in ['sfcWind', 'sfcWindmax', 'uas', 'vas']:
    print(v)
    plot_cdf_subplot(members_alps, v, ['Dual FM', 'SF 1200 km', 'SF 750 km', 'SF 300 km'], "Alps")
    plot_cdf_subplot(members_med, v, ['Dual FM', 'SF 1200 km', 'SF 750 km', 'SF 300 km'], "Med")