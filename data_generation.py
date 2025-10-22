import numpy as np
import os
import matplotlib.pyplot as plt

materials = [
    {
    "name": "Copper",
    "g_factor": 2.08,
    "line_width_range": [2.259, 2.761],
    "peak_intensity_range": [0.8, 1.2]
    },
    {   "name": "Iron",
    "g_factor": 2.10,
    "line_width_range": [0.05, 0.2],
    "peak_intensity_range": [0.1, 0.5]
    }
]

base_position = 0.5
reference_gfactor = 2.0
scaling_factor = 0.2
num_variations = 5 
noise_level = 0.05
x_axis = np.linspace(0, 1, 5000)
counter = 0

samples = []

for split in ["train", "val", "test"]:
    for material in materials:
        os.makedirs(f"dataset/{split}/{material['name']}", exist=True)

for material in materials:
    line_width_values = np.linspace(material.line_width_range[0], material.line_width_range[1], num_variations)
    peak_intensity_values = np.linspace(material.peak_intensity_range[0], material.peak_intensity_range[1], num_variations)

    for line_width in line_width_values:
        for peak_intensity in peak_intensity_values:
            counter += 1
            filename = f"{material['name']}_{counter}.png"
    
            # compute peak position with g-factor
            x_peak = base_position + (material["g_factor"] - reference_gfactor) * scaling_factor

            # generate Gaussian signal
            signal = peak_intensity * np.exp(-((x_axis - x_peak)**2) / (2 * line_width**2))

            # add small random noise
            signal += np.random.uniform(-noise_level, noise_level, size=x_axis.shape)

            # normalize signal to [0, 1]
            signal = (signal - min(signal)) / (max(signal) - min(signal))

            samples.append({
                "material": material["name"],
                "signal": signal,
                "filename": filename
            })

np.random.shuffle(samples)

num_total = len(samples)
train_end = int(num_total * 0.7)
val_end = int(num_total * 0.85)

for idx, sample in enumerate(samples):
    if idx < train_end:
        split = "train"
    elif idx < val_end:
        split = "val"
    else:
        split = "test"
    
    plt.figure(figsize=(4,4))
    plt.plot(sample["signal"])
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"dataset/{split}/{sample['material']}/{sample['filename']}", dpi=32)
    plt.close() 