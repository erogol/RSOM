import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from som import SOM


def quantization_error(som, data):
    _, distances = som.best_match(data)
    return torch.mean(torch.min(distances, dim=0)[0])


def grid_search_som(data, unit_range, epochs=1000, alpha_max=0.05, trials=3):
    results = []

    for num_units in tqdm(unit_range, desc="Grid Search"):
        trial_errors = []
        for _ in range(trials):
            som = SOM(data, num_units=num_units, alpha_max=alpha_max)
            som.train_batch(num_epoch=epochs, verbose=False)
            error = quantization_error(som, data)
            trial_errors.append(error.item())

        avg_error = np.mean(trial_errors)
        std_error = np.std(trial_errors)
        results.append((num_units, avg_error, std_error))

        print(
            f"Units: {num_units}, Avg Error: {avg_error:.4f}, Std Error: {std_error:.4f}"
        )

    return results


def find_elbow(x, y):
    # Normalize the data
    x = np.array(x)
    y = np.array(y)
    x_norm = (x - min(x)) / (max(x) - min(x))
    y_norm = (y - min(y)) / (max(y) - min(y))

    # Calculate the distances from each point to the line connecting the first and last points
    coords = np.vstack([x_norm, y_norm]).T
    first = coords[0]
    line_vec = coords[-1] - coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = coords - first
    scalar_proj = np.dot(vec_from_first, line_vec_norm)
    proj = np.outer(scalar_proj, line_vec_norm)
    distances = np.sqrt(np.sum((vec_from_first - proj) ** 2, axis=1))

    # Find the elbow point (maximum distance)
    elbow_index = np.argmax(distances)
    return x[elbow_index], y[elbow_index]


if __name__ == "__main__":
    # Load Digits dataset
    digits = load_digits()
    data = torch.from_numpy(digits.data).float()

    # Normalize the data
    data = (data - data.min()) / (data.max() - data.min())

    # Split the data into train and test sets
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

    # Define the range of units to search
    unit_range = [9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196]

    # Perform grid search
    results = grid_search_som(
        X_train, unit_range, epochs=1000, alpha_max=0.05, trials=3
    )

    # Extract units and errors
    units = [r[0] for r in results]
    errors = [r[1] for r in results]
    error_stds = [r[2] for r in results]

    # Find the elbow point
    elbow_units, elbow_error = find_elbow(units, errors)

    print(f"\nElbow point: {elbow_units:.0f} units, Error: {elbow_error:.4f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(units, errors, yerr=error_stds, fmt="o-", capsize=5)
    plt.plot(elbow_units, elbow_error, "ro", markersize=10, label="Elbow point")
    plt.xlabel("Number of Units")
    plt.ylabel("Quantization Error")
    plt.title("SOM Grid Search Results")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Train the SOM with the elbow point number of units
    best_som = SOM(data, num_units=int(elbow_units), alpha_max=0.05)
    best_som.train_batch(num_epoch=1000, verbose=True)

    # Evaluate on test set
    test_error = quantization_error(best_som, X_test)
    print(f"\nTest set quantization error: {test_error:.4f}")
