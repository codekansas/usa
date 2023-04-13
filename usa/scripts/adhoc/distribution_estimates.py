import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_distribution(num_points: int, num_samples: int, std_dev: float) -> np.ndarray:
    """Draws `num_samples` nearest neighbors from a normal distribution of `num_points` points.

    Args:
        num_points: Number of points to sample from.
        num_samples: Number of samples to draw.
        std_dev: Standard deviation of the normal distribution.

    Returns:
        Array of shape (num_samples) containing the underestimate for the final SDF.
    """

    xyz_points = np.random.normal(size=(num_points, num_samples, 3), scale=std_dev, loc=0.0)
    xyz_offsets = np.linalg.norm(xyz_points - (1.0, 0.0, 0.0), axis=-1)

    # The final SDF is the nearest neighbor, so the final SDF offset is
    # the minimum offset over every point.
    return xyz_offsets.min(0)


def main() -> None:
    num_samples = 1000
    num_points = [1, 10, 100, 1000, 10000]
    std_dev = 0.01

    # Draw the distribution for each number of points.
    distributions = [get_distribution(n, num_samples, std_dev) for n in num_points]

    # Plots the distributions on the same axes.
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    sns.set_context("paper", font_scale=1.5)
    _, ax = plt.subplots()
    for n, d in zip(num_points, distributions):
        sns.distplot(d, ax=ax, label=f"{n} points")
    ax.set_xlabel("Nearest Neighbor Distance")
    ax.set_ylabel("Density")
    ax.legend()

    plt.savefig("distribution_estimates.svg", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
