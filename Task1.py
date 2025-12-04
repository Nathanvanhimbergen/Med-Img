import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

water = loadmat('data/dataset/task_1/water/hist_140.mat')
fat = loadmat('data/dataset/task_1/fat/hist_140.mat')

# Print all top-level variable names
print(water.keys())
# water counts (1, 130) and energy (1, 131)
water_c = water['x']
water_e = water['y']

# Print all top-level variable names
print(fat.keys())
# fat counts (1, 130) and energy (1, 131)
fat_c = fat['x']
fat_e = fat['y']

# plot histograms ...
