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
        "2_5E7": {
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
        "2_5E7": {
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
    beam_fluxes = ['5E6', '1E7', '2_5E7', '5E7']

    water_counts = np.array([[np.sum(get("water", flux, "c"))] for flux in beam_fluxes])
    fat_counts = np.array([[np.sum(get("fat", flux, "c"))] for flux in beam_fluxes])

    labels = ['Total Counts and Energies']
    return water_counts, fat_counts, labels, beam_fluxes

def spectral_PCD(): 
    # Define detector energy bins: 20-50 keV and 50 keV - inf
    labels = ['20-50 keV', '>=50 keV']
    beam_fluxes = ['5E6', '1E7', '2_5E7', '5E7']

    water_counts = []
    fat_counts = []

    for flux in beam_fluxes:
        water_c = get("water", flux, "c")
        water_e = get("water", flux, "e")
        fat_c = get("fat", flux, "c")
        fat_e = get("fat", flux, "e")

        mask_20_50_w = (water_e >= 20) & (water_e < 50)
        mask_50_inf_w = (water_e >= 50)

        mask_20_50_f = (fat_e >= 20) & (fat_e < 50)
        mask_50_inf_f = (fat_e >= 50)
        
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

    return np.array(water_counts), np.array(fat_counts), labels, beam_fluxes

def plot_counts_by_flux(water_counts, fat_counts, labels, beam_fluxes, spectral):
    """
    Plot detector counts for water and fat across different beam fluxes.
    water_counts, fat_counts: shape (n_fluxes, n_bins) - counts per energy bin
    labels: energy bin labels
    beam_fluxes: list of flux values
    """
    # Use the same bar width as the spectral case
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle('Absolute Detector Counts by Beam Flux and tissue' if spectral else 'Detector Counts with spectral resolution', fontsize=16)
    
    for idx, (flux, ax) in enumerate(zip(beam_fluxes, axes)):
        w_bins = water_counts[idx]
        f_bins = fat_counts[idx]
        
        if spectral:
            x = np.arange(len(labels))
            bars_w = ax.bar(x - width/2, w_bins, width, color='C0', alpha=0.9)
            bars_f = ax.bar(x + width/2, f_bins, width, color='C1', alpha=0.9)
            ax.set_xticks(x)
        else:
            # For single bin, offset bars symmetrically around x=0
            bars_w = ax.bar(-width/2, w_bins[0], width, color='C0', alpha=0.9)
            bars_f = ax.bar(width/2, f_bins[0], width, color='C1', alpha=0.9)
            ax.set_xticks([0])
            # Make the x-range wider so bars don't fill the whole subplot
            ax.set_xlim(-1, 1)
        
        ax.set_xticklabels(labels)
        
        max_h = max(np.max(w_bins), np.max(f_bins), 1.0)
        ax.set_ylim(bottom=-0.08 * max_h)
        
        for bar in bars_w:
            center = bar.get_x() + bar.get_width() / 2
            ax.annotate('Water', xy=(center, 0), xytext=(0, -8),
                        textcoords='offset points', ha='center', va='top', fontsize=9)
        
        for bar in bars_f:
            center = bar.get_x() + bar.get_width() / 2
            ax.annotate('Fat', xy=(center, 0), xytext=(0, -8),
                        textcoords='offset points', ha='center', va='top', fontsize=9)
        
        ax.set_ylabel('Counts')
        title_suffix = 'Weighted' if spectral else 'Total'
        ax.set_title(f'{title_suffix} Detector Counts - Beam Flux: {flux} photons/s')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# Calculate contrast ratios normalized to water for both original and weighted counts
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
water_counts_s, fat_counts_s, labels_s, beam_fluxes_s = spectral_PCD()
contrast_ratio(fat_counts_s, water_counts_s, beam_fluxes_s)
plot_counts_by_flux(water_counts_s, fat_counts_s, labels_s, beam_fluxes_s, spectral=True)

# Define 
beam_area = 2 * np.pi * (5e-3)**2  # m^2