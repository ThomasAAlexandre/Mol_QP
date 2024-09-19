import imageio
import matplotlib.pyplot as plt
import numpy as np
import os


def real_vs_estimated(distances, evs, real_evs):
    """
    Plot the real and estimated dissociation curves with the correlation coefficient.

    Args:
    distances (ndarray): Array of distances between quantum systems.
    evs (ndarray): Array of estimated energy values.
    real_evs (ndarray): Array of exact energy values.
    Returns:
    None
    """
    determination_coefficient = (np.corrcoef(evs, real_evs)[0, 1]) ** 2
    correlation_text = f"R^2 = {determination_coefficient:.5f}"

    plt.scatter(distances, evs, label="Found dissociation curve", s=10)
    plt.scatter(distances, real_evs, label="Expected dissociation curve", s=10)
    plt.plot(distances, evs, linestyle="-", color="blue", alpha=0.5)
    plt.plot(distances, real_evs, linestyle="-", color="orange", alpha=0.5)
    plt.title("Found and expected dissociation curves")
    plt.xlabel("Distance (Å)")
    plt.ylabel("Energy (hartree)")
    plt.text(
        0.95,
        0.5,
        correlation_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        ha="right",
        va="center",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.legend()
    plt.show()


def create_3d(angle, tot_thetas, distances, tot_energy, angles, dists, evs):
    """
    Create a 3D scatter plot showing the relationship between angle, distance, and energy.

    Args:
    tot_thetas (List): Array containing angle parameters used in the ansatz circuit.
    distances (ndarray): Array of distances between quantum systems.
    tot_energy (List): Array of total energy values.
    angles (ndarray): Array of angle values.
    dists (List): Array of distance values.
    evs (ndarray): Array of estimated energy values.
    Returns:
    None
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(projection="3d")
    for i in range(len(tot_thetas)):
        ax1.scatter(np.abs(tot_thetas[i]), dists[i], tot_energy[i])
    ax1.scatter(np.abs(angles), distances, evs, marker="o", alpha=1)
    ax1.set_xlabel("Angle")
    ax1.set_ylabel("Distance")
    ax1.set_zlabel("Energy")
    ax1.view_init(30, angle)
    plt.draw()


def create_3d_rotating(distances, tot_thetas, tot_energy, angles_theta, dists, evs):
    """
    Create a rotating 3D plot to visualize the relationship between angle, distance, and energy.

    Args:
    tot_thetas (List): Array containing angle parameters used in the ansatz circuit.
    distances (ndarray): Array of distances between quantum systems.
    tot_energy (List): Array of total energy values.
    angles (ndarray): Array of angle values.
    dists (List): Array of distance values.
    evs (ndarray): Array of estimated energy values.
    Returns:
    None
    """
    angles = np.linspace(0, 360 - (360 / 90), 90)
    filenames = []
    for angle in angles:
        create_3d(angle, tot_thetas, distances, tot_energy, angles_theta, dists, evs)
        filename = f"frame_{int(angle)}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    with imageio.get_writer(
        "rotating_3d_plot.gif", mode="I", duration=1, loop=0
    ) as writer:
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
            os.remove(filename)


def process_results_f_plot(
    result,
    distance,
    energies,
    thetas,
    angles,
    dists,
    tot_thetas,
    tot_energy,
    file_no,
    nuclear_repulsion_energy,
):
    """
    Process the optimization parameters and results and append necessary values to respective lists
    used for plots.

    Args:
    result (OptimizeResult): Optimization result object.
    distance (float): Distance between quantum systems.
    energies (list): List of energy values.
    thetas (list): List of angle values.
    angles (NDArray): List of angle values to be updated.
    dists (list): List of distance values to be updated.
    tot_thetas (list): List of angle parameters used in the ansatz circuit to be updated.
    tot_energy (list): List of total energy values to be updated.
    file_no (int): The number of the file
    nuclear_repulsion_energy (float): Nuclear repulsion energy

    Returns:
    None
    """
    angles[file_no] = result.x[0]
    dists.append(np.repeat(distance, len(energies)))
    tot_thetas.append(np.array(thetas).flatten())
    tot_energy.append(energies)
    for i in range(len(energies)):
        energies[i] += nuclear_repulsion_energy
