import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# load data
water_1E7_data = np.load('data/dataset/task_2/water/rate_1E7.npy')
water_2_5E7_data = np.load('data/dataset/task_2/water/rate_2.5E7.npy')
water_5E6_data = np.load('data/dataset/task_2/water/rate_5E6.npy')
water_5E7_data = np.load('data/dataset/task_2/water/rate_5E7.npy')

fat_1E7_data = np.load('data/dataset/task_2/fat/rate_1E7.npy')
fat_2_5E7_data = np.load('data/dataset/task_2/fat/rate_2.5E7.npy')
fat_5E6_data = np.load('data/dataset/task_2/fat/rate_5E6.npy')
fat_5E7_data = np.load('data/dataset/task_2/fat/rate_5E7.npy')

data = {
    "water": {
        "5E6": {
            "c": water_5E6_data[1],
            "e": water_5E6_data[0] * 1000, # convert to keV
        },
        "1E7": {
            "c": water_1E7_data[1],
            "e": water_1E7_data[0] * 1000, # convert to keV
        },
        "2.5E7": {
            "c": water_2_5E7_data[1],
            "e": water_2_5E7_data[0] * 1000, # convert to keV
        },
        "5E7": {
            "c": water_5E7_data[1],
            "e": water_5E7_data[0] * 1000, # convert to keV
        },
    },
    "fat": {
        "5E6": {
            "c": fat_5E6_data[1],
            "e": fat_5E6_data[0] * 1000, # convert to keV
        },
        "1E7": {
            "c": fat_1E7_data[1],
            "e": fat_1E7_data[0] * 1000, # convert to keV
        },
        "2.5E7": {
            "c": fat_2_5E7_data[1],
            "e": fat_2_5E7_data[0] * 1000, # convert to keV
        },
        "5E7": {
            "c": fat_5E7_data[1],
            "e": fat_5E7_data[0] * 1000, # convert to keV
        },
    },
}
def get(material, flux, kind):
    return data[material][flux][kind] # x = get("water", "1E7", "c")
# print('data[0] = ', water_1E7_data[0])

# print('data[1] =',water_1E7_data[1])

# x = get("water", "5E7", "e")
# y = get("water", "5E7", "c")
# plt.plot(x,y)
# plt.grid()
# plt.title('Water Spectrum at 1E7 photons/s')
# plt.xlabel('Energy (keV)')
# plt.ylabel('Counts')
# plt.show()
def pure_PCD():
    beam_fluxes = ['5E6', '1E7', '2.5E7', '5E7']

    water_counts = np.array([[np.sum(get("water", flux, "c"))] for flux in beam_fluxes])
    fat_counts = np.array([[np.sum(get("fat", flux, "c"))] for flux in beam_fluxes])

    labels = ['Total Counts and Energies']
    return water_counts, fat_counts, labels, beam_fluxes

def spectral_PCD(): 
    # Define detector energy bins: 20-50 keV and 50 keV - inf
    labels = ['20-50 keV', '>=50 keV']
    beam_fluxes = ['5E6', '1E7', '2.5E7', '5E7']

    water_counts = []
    fat_counts = []
    water_masks = []
    fat_masks = []

    for flux in beam_fluxes:
        water_c = get("water", flux, "c")
        water_e = get("water", flux, "e")
        fat_c = get("fat", flux, "c")
        fat_e = get("fat", flux, "e")

        mask_20_50_w = (water_e >= 20) & (water_e < 50)
        mask_50_inf_w = (water_e >= 50)

        mask_20_50_f = (fat_e >= 20) & (fat_e < 50)
        mask_50_inf_f = (fat_e >= 50)
        
        water_masks.append((mask_20_50_w, mask_50_inf_w))
        fat_masks.append((mask_20_50_f, mask_50_inf_f))
        
        # ======= weighted counts =======
        water_counts_bins = np.array([
            np.sum(water_c[mask_20_50_w] * 0.8), # 80% weight for 20-50 keV
            np.sum(water_c[mask_50_inf_w] * 0.2) # 20% weight for >=50 keV
        ])
        fat_counts_bins = np.array([
            np.sum(fat_c[mask_20_50_f] * 0.8),
            np.sum(fat_c[mask_50_inf_f] * 0.2)
        ])

        water_counts.append(water_counts_bins)
        fat_counts.append(fat_counts_bins)

    return np.array(water_counts), np.array(fat_counts), labels, beam_fluxes, water_masks, fat_masks


def plot_counts_by_flux(water_counts, fat_counts, labels, beam_fluxes, spectral, water_masks=None, fat_masks=None):
    """
    Plot detector counts for water and fat across different beam fluxes.
    water_counts, fat_counts: shape (n_fluxes, n_bins) - counts per energy bin
    labels: energy bin labels
    beam_fluxes: list of flux values
    water_masks, fat_masks: optional masks for weighted spectrum plotting
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle('Absolute Detector Counts by Beam Flux and tissue' if spectral else 'Detector Counts without spectral resolution', fontsize=16)

    for idx, (flux, ax) in enumerate(zip(beam_fluxes, axes)):
        w_bins = water_counts[idx]
        f_bins = fat_counts[idx]
        
        # Get raw energy spectra using flux from beam_fluxes
        water_e = get("water", flux, "e")
        water_c = get("water", flux, "c")
        fat_e = get("fat", flux, "e")
        fat_c = get("fat", flux, "c")

        if spectral and water_masks is not None and fat_masks is not None:
            # Apply weights to spectra for plotting
            water_c_weighted = water_c.copy()
            fat_c_weighted = fat_c.copy()
            
            mask_20_50_w, mask_50_inf_w = water_masks[idx]
            mask_20_50_f, mask_50_inf_f = fat_masks[idx]
            
            water_c_weighted[mask_20_50_w] *= 0.8
            water_c_weighted[mask_50_inf_w] *= 0.2
            fat_c_weighted[mask_20_50_f] *= 0.8
            fat_c_weighted[mask_50_inf_f] *= 0.2

        if spectral:
            # Bar width based on energy ranges
            width_20_50 = 15    # narrower bars for 20-50 keV range
            width_50_inf = 15   # narrower bars for >=50 keV
            
            # Position bars for 20-50 keV bin (two bars side-by-side)
            bars_w = ax.bar(35 - width_20_50/2, w_bins[0], width_20_50, color='C0', alpha=0.7, edgecolor='black', label='Water')
            bars_f = ax.bar(35 + width_20_50/2, f_bins[0], width_20_50, color='C1', alpha=0.6, edgecolor='black', label='Fat')
            
            # Position bars for >=50 keV bin (two bars side-by-side, starting at 50 keV)
            bars_w2 = ax.bar(50 + width_50_inf/2, w_bins[1], width_50_inf, color='C0', alpha=0.7, edgecolor='black')
            bars_f2 = ax.bar(50 + width_50_inf/2 + width_50_inf, f_bins[1], width_50_inf, color='C1', alpha=0.6, edgecolor='black')
            
            ax.set_xlim(0, 100)
            # Plot weighted raw spectra as overlay (on secondary y-axis for clarity)
            ax2 = ax.twinx()
            ax2.plot(water_e, water_c_weighted, color='C0', alpha=0.9, linewidth=2)
            ax2.plot(fat_e, fat_c_weighted, color='C1', alpha=0.9, linewidth=2)
            ax2.set_yticks([])  # Hide y-axis ticks
            ax2.set_ylabel('')  # Hide y-axis label
            ax2.set_xticks([20, 50, 90])  # Only show these x-ticks
            ax2.set_xticklabels(['20', '50 keV', '->\u221E'])
            ax2.spines['right'].set_visible(False)  # Hide right spine
            ax2.set_xlim(0, 100)
            
            # Add annotations
            ax.text(34, -0.10 * ax.get_ylim()[1], 'Water', ha='center', fontsize=10, weight='bold')
            ax.text(66, -0.10 * ax.get_ylim()[1], 'Fat', ha='center', fontsize=10, weight='bold')
        else:
            # For single bin, position bar at 50 keV (split between left/right)
            width_bar = 10
            bars_w = ax.bar(45, w_bins[0], width_bar, color='C0', alpha=0.7, edgecolor='black')
            bars_f = ax.bar(55, f_bins[0], width_bar, color='C1', alpha=0.6, edgecolor='black')
            
            ax.set_xlim(25, 75)
            ax2 = ax.twinx()
            ax2.plot(33.3333+water_e/3, water_c, color='C0', alpha=0.9, linewidth=2)
            ax2.plot(33.3333+fat_e/3, fat_c, color='C1', alpha=0.9, linewidth=2)
            ax2.set_xlim(0, 120)
            ax2.set_yticks([])  # Hide y-axis ticks
            ax2.set_ylabel('')  # Hide y-axis label
            ax2.set_xticks([40, 50, 70])  # Only show these x-ticks
            ax2.set_xticklabels(['20', '50 keV', '->\u221E'])
            ax2.spines['right'].set_visible(False)  # Hide right spine
            ax2.set_xlim(25, 75)
            # Add annotations
            ax.text(45, -0.10 * ax.get_ylim()[1], 'Water', ha='center', fontsize=10, weight='bold')
            ax.text(55, -0.10 * ax.get_ylim()[1], 'Fat', ha='center', fontsize=10, weight='bold')

        # Mark 50 keV boundary with vertical line
        # ax.axvline(x=50, color='red', linestyle='--', linewidth=2.5, label='50 keV boundary', zorder=5)
        
        title_suffix = 'Weighted' if spectral else 'Total'
        ax.set_title(f'{title_suffix} Detector Counts - Beam Flux: {flux} photons/s')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def contrast_ratio(tissue_counts, water_counts, beam_fluxes):
    """
    Calculate contrast ratio: (tissue - water) / water * 100
    tissue_counts, water_counts: shape (n_fluxes, n_bins)
    beam_fluxes: list of flux values for labeling
    """
    contrast = np.zeros_like(tissue_counts)
    
    for i in range(len(water_counts)):
        for j in range(tissue_counts.shape[1]):
            if water_counts[i, j] != 0:
                contrast[i, j] = (tissue_counts[i, j] - water_counts[i, j]) * 100 / water_counts[i, j]
                print(f'Contrast ratio - Flux {beam_fluxes[i]}, Bin {j}: {contrast[i, j]:.2f}%')
    print('\n\n')
    return contrast

# Run analysis and plot pure PCD
water_counts_p, fat_counts_p, labels_p, beam_fluxes_p = pure_PCD()
contrast_ratio(fat_counts_p, water_counts_p, beam_fluxes_p)
plot_counts_by_flux(water_counts_p, fat_counts_p, labels_p, beam_fluxes_p, spectral=False)

# Run analysis and plot spectral PCD
water_counts_s, fat_counts_s, labels_s, beam_fluxes_s, water_masks_s, fat_masks_s = spectral_PCD()
contrast_ratio(fat_counts_s, water_counts_s, beam_fluxes_s)
plot_counts_by_flux(water_counts_s, fat_counts_s, labels_s, beam_fluxes_s, spectral=True, water_masks=water_masks_s, fat_masks=fat_masks_s)

