import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
#=============================================================
# load data into 4x8 matrix 
# every 2 columns is one dataset: energies, counts respectively
# Every row is a different detector pixel size (200um, 250um, 500um, 1000um)
#=============================================================
targets = ["I0_rate_1.32E7.npy", "I0_rate_2.64E7.npy", "I0_rate_6.60E7.npy", "I0_rate_1.32E8.npy"]
sizes = ["200um", "250um", "500um", "1000um"]
matrix = np.empty((len(sizes), len(targets)*2), dtype=object)

for i, size in enumerate(sizes):
    for j, target in enumerate(targets):
        data = np.load(r'data/dataset/task_3/' + size + '/' + target)
        matrix[i,j*2] = data[0]*1000  # convert to keV
        matrix[i,j*2+1] = data[1]  # counts


# #===========================================================
# # Line graphs for all sizes and fluence rates
# #===========================================================
# fig, ax = plt.subplots(len(sizes), len(targets), figsize=(14,10))

# for i, size in enumerate(sizes):
#     for j, target in enumerate(targets):
#         energy = matrix[i,j*2]
#         counts = matrix[i,j*2+1]

#         counts_smooth = gaussian_filter1d(counts, sigma=1)
        
#         #ax[i,j].bar(energy, counts, width=1, color='C0', edgecolor='b', alpha=0.8)
#         ax[i,j].plot(energy, counts_smooth, color='C0', alpha=0.8)
#         ax[i,j].set_yscale('log')
#         ax[i,j].set_ylim(0.5, 6*10**6)
#         ax[i,j].set_title(f'Size: {size}, Rate: {target.split("_")[-1][:-4]}')
#         ax[i,j].set_xlabel('Energy (keV)')
#         ax[i,j].set_ylabel('Counts')
#         ax[i,j].grid(linestyle='--', alpha=0.4)

# plt.tight_layout()
# plt.show()

# #===========================================================
# # Histogram plot for a single size and fluence rate
# #===========================================================
# plt.figure(figsize=(10,5))
# #plt.bar(matrix[0,0], matrix[0,1], width=1, label="200um", edgecolor='b', linewidth=0.5)
# # apply gaussian smoothing
# counts_smooth = gaussian_filter1d(matrix[0,1], sigma=1)
# plt.plot(matrix[0,0], counts_smooth, label=sizes[0], alpha=0.8, linewidth=2)
# plt.yscale('log')
# plt.xlabel('Energy (keV)', fontsize=12)
# plt.ylabel('Counts (log scale)', fontsize=12)
# plt.title(f'Logarithmic Detector Counts (smoothed) of Size: {sizes[0]} and Fluence Rate: {targets[0].split("_")[-1][:-4]}', fontsize=12)
# plt.grid(True, alpha=0.3, which='both', linestyle='--')
# plt.xticks(np.arange(0, 350, 25))
# plt.xlim(15, 305)
# plt.ylim(0.5, 6*10**6)
# #plt.legend(fontsize=11)
# plt.tight_layout()
# plt.show()

#===========================================================
# Line plots for a single fluence rate with different sizes 
#===========================================================
# plt.figure(figsize=(10, 5))
# for i, size in enumerate(sizes):
#     energy = np.asarray(matrix[i, 2]).ravel()
#     counts = np.asarray(matrix[i, 3]).ravel()
    
#     # apply gaussian smoothing
#     counts_smooth = gaussian_filter1d(counts, sigma=1)
    
#     plt.plot(energy, counts_smooth, label=size, alpha=0.8, linewidth=2)

# plt.yscale('log')
# plt.xlabel('Energy (keV)', fontsize=12)
# plt.ylabel('Counts (log scale)', fontsize=12)
# plt.title(f'Logarithmic Detector Counts (Smoothed) of Fluence Rate: {targets[2].split("_")[-1][:-4]} With Different Pixel Sizes', fontsize=13)
# plt.grid(True, alpha=0.3, which='both', linestyle='--')
# plt.xticks(np.arange(0, 350, 25))
# plt.xlim(15, 305)
# plt.ylim(0.5, 6*10**6)
# plt.legend(fontsize=11)
# plt.tight_layout()
# plt.show()

#===========================================================
# Line plots for a single size with different fluence rates
#===========================================================
# plt.figure(figsize=(10, 5))
# for i, target in enumerate(targets):
#     energy = np.asarray(matrix[1, i*2]).ravel()
#     counts = np.asarray(matrix[1, i*2+1]).ravel()
    
#     # apply gaussian smoothing
#     counts_smooth = gaussian_filter1d(counts, sigma=1)
    
#     plt.plot(energy, counts_smooth, label=target.split("_")[-1][:-4], alpha=0.8, linewidth=2)

# plt.yscale('log')
# plt.xlabel('Energy (keV)', fontsize=12)
# plt.ylabel('Counts (log scale)', fontsize=12)
# plt.title(f'Logarithmic Detector Counts (Smoothed) of Size: {sizes[1]} With Different Fluence Rates', fontsize=13)
# plt.grid(True, alpha=0.3, which='both', linestyle='--')
# plt.xticks(np.arange(0, 350, 25))
# plt.xlim(15, 305)
# plt.ylim(0.5, 6*10**6)
# plt.legend(fontsize=11)
# plt.tight_layout()
# plt.show()

#===========================================================
# Line plots for with constant sizes showing different fluence rates
#===========================================================
# fig, ax = plt.subplots(2, 2, figsize=(10, 5))

# for j, size in enumerate(sizes):
#     row = j // 2
#     col = j % 2
#     for i, target in enumerate(targets):
#         energy = np.asarray(matrix[j, i*2]).ravel()
#         counts = np.asarray(matrix[j, i*2+1]).ravel()
        
#         # apply gaussian smoothing
#         counts_smooth = gaussian_filter1d(counts, sigma=1)
        
#         ax[row, col].plot(energy, counts_smooth, label=target.split("_")[-1][:-4], alpha=0.8)
    
#     ax[row, col].set_yscale('log')
#     ax[row, col].set_xlabel('Energy (keV)', fontsize=12)
#     ax[row, col].set_ylabel('Counts (log scale)', fontsize=12)
#     ax[row, col].set_title(f'Size: {size}', fontsize=13)
#     ax[row, col].grid(True, alpha=0.3, which='both', linestyle='--')
#     ax[row, col].set_xticks(np.arange(0, 350, 25))
#     ax[row, col].set_xlim(15, 305)
#     ax[row, col].set_ylim(0.5, 6*10**6)
#     ax[row, col].legend(loc='upper right', fontsize=6)

# plt.tight_layout()
# plt.show()

#===========================================================
# Line plots with constant fluence rate showing different pixel sizes
#===========================================================
# fig, ax = plt.subplots(2, 2, figsize=(10, 5))

# for i, target in enumerate(targets):
#     row = i // 2
#     col = i % 2
#     for j, size in enumerate(sizes):
#         # Obtain energy and counts for current size and target
#         energy = np.asarray(matrix[j, i*2]).ravel()
#         counts = np.asarray(matrix[j, i*2+1]).ravel()
        
#         # apply gaussian smoothing
#         counts_smooth = gaussian_filter1d(counts, sigma=1)
        
#         ax[row, col].plot(energy, counts_smooth, label=size, alpha=0.8)
    
#     ax[row, col].set_yscale('log')
#     ax[row, col].set_xlabel('Energy (keV)', fontsize=12)
#     ax[row, col].set_ylabel('Counts (log scale)', fontsize=12)
#     ax[row, col].set_title(f'Rate: {target.split("_")[-1][:-4]}', fontsize=13)
#     ax[row, col].grid(True, alpha=0.3, which='both', linestyle='--')
#     ax[row, col].set_xticks(np.arange(0, 350, 25))
#     ax[row, col].set_xlim(15, 305)
#     ax[row, col].set_ylim(0.5, 6*10**6)
#     ax[row, col].legend(loc='upper right', fontsize=6)

# plt.tight_layout()
# plt.show()