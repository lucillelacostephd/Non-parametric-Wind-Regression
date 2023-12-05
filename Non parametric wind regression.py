# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:02:23 2023

@author: lb945465
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Load the data
data = pd.read_csv(r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Spyder\1_Preparation of the Database\PMF_DNIC\CBPF_merged.csv',
                   index_col="Date")

# Define wind direction bins and labels
wind_direction_bins = np.arange(0, 361, 10)  # 0, 10, ..., 350, 360
wind_direction_labels = np.arange(5, 361, 10)  # 5, 15, ..., 355, 365 (midpoints of bins)

# Bin the wind direction data 
if 'wd_bin' not in data.columns:
    data['wd_bin'] = pd.cut(data['wd'], bins=wind_direction_bins, labels=wind_direction_labels, right=False)

# Define the colors for specific column names (for consistency)
color_dict = {
    "Fuel evaporation": "#1f77b4",   # blue
    "Combustion": "#ff7f0e",   # orange
    "Natural gas": "#2ca02c",   # green
    "Diesel traffic": "#d62728",   # red
    "Industrial solvents": "#9467bd",   # purple
    "Gasoline traffic": "#8c564b",   # brown
    "Biogenic": "#7f7f7f",   # gray
    # The other factors are not present in the current dataset, so they are commented out
    # "Urban mix": "#e377c2",   # pink
    # "Factor 1": "#bcbd22",   # lime green
    # "Factor 2": "#17becf",  # teal
    # "Factor 3": "#aec7e8",  # light blue
    # "Factor 4": "#ffbb78",  # peach
    # "Factor 5": "#98df8a",  # light green
    # "Factor 6": "#ff9896",  # salmon
    # "Factor 7": "#c5b0d5"   # lavender
}

# Select the sources to plot
sources_to_plot = ['Gasoline traffic', 'Biogenic', 'Industrial solvents', 'Fuel evaporation', 'Diesel traffic', 'Combustion', 'Natural gas']

# Calculate the mean concentration for each source within each wind direction bin
source_means_by_wd = data.groupby('wd_bin')[sources_to_plot].mean()

# Apply LOWESS smoothing to each source
smoothed_data = {}
for source in sources_to_plot:
    # Apply LOWESS smoothing, which returns an array of the same length as the input data
    x_values = source_means_by_wd.index.astype(int)
    smoothed_data[source] = lowess(source_means_by_wd[source], x_values, frac=0.2, it=0, return_sorted=False)

# Plot the smoothed data for each source using the specified colors
plt.figure(figsize=(8, 4), dpi=300)
for source in sources_to_plot:
    plt.plot(x_values, smoothed_data[source], label=source, color=color_dict[source])

# Create a lineplot comparison
plt.title('LOWESS Smoothed Mean Concentration of Various Sources by Wind Direction')
plt.xlabel('Wind Direction (°N)')
plt.ylabel('Smoothed Mean Concentration')
plt.xticks(np.arange(0, 361, 45))
plt.xlim(0, 360)  # Set the x-limits to cover the full range of wind directions
plt.grid(True)
plt.legend()
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()  # Display the plot
