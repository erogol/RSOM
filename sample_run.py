import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from rsom import RSOM


def get_node_coordinates(som, pca):
    coords = []
    for i in range(som.height):
        for j in range(som.width):
            node_index = i * som.width + j
            node_weights = som.W[node_index].detach().numpy()
            coord = pca.transform([node_weights])[0]
            coords.append(coord)
    return np.array(coords)


# Load Iris dataset
data = load_digits().data
data = torch.from_numpy(data).float()
print(data.shape)

# Initialize SOM
som = RSOM(data, alpha_max=0.05, num_units=49)

# Train SOM
som.train_batch(num_epoch=1000, verbose=True)

# Get salient instances and units
salient_insts = som.salient_insts()
salient_units = som.salient_units()

# Perform PCA to reduce data to 2D for visualization
pca = PCA(n_components=2)
data_2d = pca.fit_transform(som.X.numpy())
units_2d = pca.transform(som.W.detach().numpy())

# Get node coordinates
node_coords = get_node_coordinates(som, pca)

# Create a plot
plt.figure(figsize=(12, 8))

# Plot data points
salient_mask = som.inst_saliency.numpy()
plt.scatter(
    data_2d[salient_mask, 0],
    data_2d[salient_mask, 1],
    c=som.ins_unit_assign[salient_mask],
    cmap="viridis",
    alpha=0.6,
    label="Salient Samples",
)
plt.scatter(
    data_2d[~salient_mask, 0],
    data_2d[~salient_mask, 1],
    c="red",
    marker="x",
    alpha=0.6,
    label="Outlier Samples",
)

# Plot SOM units
salient_units_mask = som.unit_saliency.numpy()
plt.scatter(
    node_coords[salient_units_mask, 0],
    node_coords[salient_units_mask, 1],
    c="black",
    marker="s",
    s=50,
    label="Salient Units",
)
plt.scatter(
    node_coords[~salient_units_mask, 0],
    node_coords[~salient_units_mask, 1],
    c="red",
    marker="s",
    s=50,
    label="Outlier Units",
)

# Draw lattice lines
for i in range(som.height):
    for j in range(som.width):
        node_index = i * som.width + j
        if j < som.width - 1:  # Horizontal line
            next_node_index = node_index + 1
            plt.plot(
                [node_coords[node_index, 0], node_coords[next_node_index, 0]],
                [node_coords[node_index, 1], node_coords[next_node_index, 1]],
                "gray",
                alpha=0.5,
            )
        if i < som.height - 1:  # Vertical line
            next_node_index = node_index + som.width
            plt.plot(
                [node_coords[node_index, 0], node_coords[next_node_index, 0]],
                [node_coords[node_index, 1], node_coords[next_node_index, 1]],
                "gray",
                alpha=0.5,
            )

# Add labels and title
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("SOM Units and Data Samples with Outliers and Lattice")
plt.legend()

# Show the plot
plt.show()

# Optional: Print some statistics
print(f"Number of salient samples: {salient_mask.sum()}")
print(f"Number of outlier samples: {(~salient_mask).sum()}")
print(f"Number of salient units: {salient_units_mask.sum()}")
print(f"Number of outlier units: {(~salient_units_mask).sum()}")

# Create a new figure for the perfect 2D lattice plot
plt.figure(figsize=(12, 12))

# Create a perfect 2D grid for SOM nodes
grid_x, grid_y = np.meshgrid(np.arange(som.width), np.arange(som.height))
grid_x = grid_x.flatten()
grid_y = grid_y.flatten()

# Plot the perfect grid
plt.scatter(grid_x, grid_y, c="lightgray", s=200, marker="s")

# Draw grid lines
for x in range(som.width):
    plt.axvline(x, color="lightgray", linestyle="--")
for y in range(som.height):
    plt.axhline(y, color="lightgray", linestyle="--")

# Get the unit assignments for each sample
unit_assignments = som.ins_unit_assign.numpy()

# Calculate the positions of samples on the grid
sample_x = grid_x[unit_assignments].astype(float)
sample_y = grid_y[unit_assignments].astype(float)

# Add some jitter to prevent complete overlap
jitter = 0.2
sample_x += np.random.uniform(-jitter, jitter, sample_x.shape)
sample_y += np.random.uniform(-jitter, jitter, sample_y.shape)

# Plot the samples on the grid
scatter = plt.scatter(
    sample_x, sample_y, c=som.ins_unit_assign, cmap="viridis", alpha=0.6
)

# Highlight outlier samples
outlier_mask = ~som.inst_saliency.numpy()
plt.scatter(
    sample_x[outlier_mask],
    sample_y[outlier_mask],
    facecolors="none",
    edgecolors="red",
    s=50,
    linewidths=2,
)

# Highlight outlier units
for unit in np.where(~som.unit_saliency.numpy())[0]:
    unit_x, unit_y = som.unit_cords(unit)
    plt.gca().add_patch(
        plt.Circle((unit_x, unit_y), 0.4, fill=False, edgecolor="red", linewidth=2)
    )

# Set labels and title
plt.xlabel("SOM Width")
plt.ylabel("SOM Height")
plt.title("Samples Mapped to Perfect 2D SOM Lattice")

# Set tick labels
plt.xticks(range(som.width))
plt.yticks(range(som.height))

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("Unit Assignment")

# Adjust plot limits
plt.xlim(-0.5, som.width - 0.5)
plt.ylim(-0.5, som.height - 0.5)

# Show the plot
plt.tight_layout()
plt.show()

# Create a folder to save outlier images
output_folder = "outlier_digits"
os.makedirs(output_folder, exist_ok=True)

# Get the original digit images and their labels
digits = load_digits()
images = digits.images
labels = digits.target

# Find the indices of outlier samples
outlier_indices = np.where(~salient_mask)[0]

# Save outlier images
for i, idx in enumerate(outlier_indices):
    img = images[idx]
    label = labels[idx]

    # Normalize the image to 0-255 range
    img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(
        np.uint8
    )

    # Create a PIL Image
    pil_img = Image.fromarray(img_normalized)

    # Save the image
    filename = f"outlier_{i}_label_{label}.png"
    pil_img.save(os.path.join(output_folder, filename))

print(f"Saved {len(outlier_indices)} outlier images to '{output_folder}' folder.")

# Find samples closest to salient units
salient_folder = "salient_digits"
os.makedirs(salient_folder, exist_ok=True)
salient_unit_indices = np.where(som.unit_saliency.numpy())[0]

for i, unit_idx in enumerate(salient_unit_indices):
    # Find the sample closest to this salient unit
    unit_weights = som.W[unit_idx].detach().numpy()
    distances = np.linalg.norm(data.numpy() - unit_weights, axis=1)
    closest_sample_idx = np.argmin(distances)

    img = images[closest_sample_idx]
    label = labels[closest_sample_idx]

    # Normalize the image to 0-255 range
    img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(
        np.uint8
    )

    # Create a PIL Image
    pil_img = Image.fromarray(img_normalized)

    # Save the image
    filename = f"salient_unit_{i}_label_{label}.png"
    pil_img.save(os.path.join(salient_folder, filename))

print(
    f"Saved {len(salient_unit_indices)} salient unit images to '{salient_folder}' folder."
)
