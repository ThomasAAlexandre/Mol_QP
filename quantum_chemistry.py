from quantum_chemistry_utils import (
    annihilation_operators_with_jordan_wigner,
    minimize_expectation_value,
    build_qubit_hamiltonian,
    exact_minimal_eigenvalue,
    ansatz,
)
from quantum_chemistry_plot import process_results_f_plot

from scipy.optimize import minimize

import os
import numpy as np


def get_evs():
    """
    Compute the minimal energy of electrons in a molecule.

    Args:
    None
    Returns:
    distances (ndarray): Array of distances between quantum systems.
    angles (ndarray): Array of angle values.
    evs (ndarray): Array of estimated energy values.
    real_evs (ndarray): Array of exact energy values.
    dists (List): List by distance of ndarrays containing distance values repeated per minimization step.
    tot_thetas (List): List by distance of ndarrays containing angle parameters used in the ansatz circuit.
    tot_energy (List): List by distance of lists containing each energy value at every step.
    """

    files = os.listdir("h2_mo_integrals")
    datapath = "h2_mo_integrals/"

    distances, real_evs, evs, angles = (np.zeros(len(files)) for _ in range(4))
    dists, tot_thetas, tot_energy = ([] for _ in range(3))

    for file_no, file in enumerate(files):

        file_path = os.path.join(datapath, file)

        npzfile = np.load(file_path)

        distance = npzfile["distance"]
        distances[file_no] = distance
        one_body = npzfile["one_body"]
        two_body = npzfile["two_body"]
        nuclear_repulsion_energy = npzfile["nuclear_repulsion_energy"]

        nb_orbitals = one_body.shape[0]

        annihilators = annihilation_operators_with_jordan_wigner(nb_orbitals)
        creators = [annihilator.adjoint() for annihilator in annihilators]

        ansatz_fct = ansatz(nb_orbitals)

        hamilt = build_qubit_hamiltonian(one_body, two_body, annihilators, creators)
        result, energies, thetas = minimize_expectation_value(
            hamilt,
            ansatz_fct,
            starting_params=[0],
            minimizer=minimize,
            execute_opts={"shots": 1000},
        )

        evs[file_no] = np.array(result.fun) + nuclear_repulsion_energy
        real_evs[file_no] = (
            exact_minimal_eigenvalue(hamilt) + nuclear_repulsion_energy
        ).real

        process_results_f_plot(
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
        )

    return distances, tot_thetas, tot_energy, angles, dists, evs, real_evs
