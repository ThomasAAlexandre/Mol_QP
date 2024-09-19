from typing import List, Callable

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
from scipy.optimize import OptimizeResult


def annihilation_operators_with_jordan_wigner(num_states: int) -> List[SparsePauliOp]:
    """
    Builds the annihilation operators as sum of two Pauli Strings for given number of fermionic
    states using the Jordan Wigner mapping.

    Args:
    num_states (int): Number of fermionic states.
    Returns:
    List[SparsePauliOp]: The annihilation operators.
    """

    z_bits = np.tril(np.ones((num_states, num_states)), -1)
    x_bits = np.eye(num_states)
    paulis1 = PauliList.from_symplectic(z_bits, x_bits)
    paulis2 = PauliList.from_symplectic(z_bits + np.eye(num_states), x_bits)

    annihilation_operators = []
    for i in range(num_states):
        annihilation_operators.append(
            SparsePauliOp([paulis1[i], paulis2[i]], [0.5, 0.5j])
        )

    return annihilation_operators


def build_qubit_hamiltonian(
    one_body: NDArray[np.complex_],
    two_body: NDArray[np.complex_],
    annihilation_operators: List[SparsePauliOp],
    creation_operators: List[SparsePauliOp],
) -> SparsePauliOp:
    """
    Build a qubit Hamiltonian from the one body and two body fermionic Hamiltonians.
    Args:
    one_body (NDArray[np.complex_]): The matrix for the one body Hamiltonian
    two_body (NDArray[np.complex_]): The array for the two body Hamiltonian
    annihilation_operators (List[SparsePauliOp]): List of sums of two Pauli strings
    creation_operators (List[SparsePauliOp]): List of sums of two Pauli strings (adjoint of
    annihilation_operators)
    Returns:
    SparsePauliOp: The total Hamiltonian as a sum of Pauli strings

    """
    qubit_hamiltonian = SparsePauliOp(["IIII"], coeffs=[0])
    for i in range(len(annihilation_operators)):
        for j in range(len(annihilation_operators)):
            creator_annihilator = creation_operators[i].compose(annihilation_operators[j])
            qubit_hamiltonian += one_body[i,j] * creator_annihilator
            for k in range(len(annihilation_operators)):
                creator_annihilator_k = ((creation_operators[i].compose(creation_operators[j]))
                                         .compose(annihilation_operators[k]))
                for l in range(len(annihilation_operators)):
                    qubit_hamiltonian += ((1 / 2) * two_body[i, j, k, l] *
                                          (creator_annihilator_k.compose(annihilation_operators[l])))

    return qubit_hamiltonian.simplify()


def minimize_expectation_value(
    observable: SparsePauliOp,
    ansatz: QuantumCircuit,
    starting_params: list,
    minimizer: Callable,
    execute_opts: dict = {},
) -> OptimizeResult:
    """
    Uses the minimizer to search for the minimal expectation value of the observable for the
    state that the ansatz produces given some parameters.

    Args:
    observable (SparsePauliOp): The observable which the expectation value will be
    minimized.
    ansatz (QuantumCircuit): A paramterized quantum circuit used to produce quantum state.
    starting_params (list): The initial parameter of the circuit used to start the
    minimization.
    backend (Backend): A Qiskit backend on which the circuit will be executed.
    minimizer (Callable): A callable function, based on scipy.optimize.minimize which only
    takes a function and starting params as inputs.
    execute_opts (dict, optional): Options to be passed to the Qiskit execute function.

    Returns:
    optimize_result (OptimizeResult): The result of the optimization
    energies (list[float]): List of energies obtained during optimization.
    thetas (list[]): List of parameter values corresponding to each energy evaluation.
    """

    energies, thetas = [], []
    estimator = Estimator()

    def cost_function(params) -> float:
        """
        Computes the energy of the observable for a quantum state given by the ansatz
        with specific parameters.

        Args:
        params: parameter of the circuit used for the minimization.
        Returns:
        Energy (float): The energy expectation value of the observable for the given state.
        """
        state_qc = ansatz.bind_parameters(params)
        job = estimator.run(state_qc, observable, shots=execute_opts["shots"])
        energy = job.result().values[0]

        energies.append(energy)
        thetas.append(params)
        return energy

    optimize_result = minimizer(cost_function, starting_params, method="COBYLA")

    return optimize_result, energies, thetas


def exact_minimal_eigenvalue(observable: SparsePauliOp) -> float:
    """
    Computes the minimal eigenvalue of an observable.

    Args:
    observable (SparsePauliOp): The observable to diagonalize.
    Returns:
    float: The minimal eigenvalue of the observable.
    """
    minimal_eigenvalue = min(np.linalg.eigvals(observable.to_matrix()))

    return minimal_eigenvalue


def ansatz(nb_orbitals: int) -> QuantumCircuit:
    """
    Construct the ansatz quantum circuit for H2

    Args:
    nb_orbitals (int): The number of orbitals in the system.
    Returns:
    qc (QuantumCircuit): The ansatz quantum circuit.
    """
    qc = QuantumCircuit(nb_orbitals)
    qc.x(0)
    theta_param = Parameter("theta")
    qc.ry(theta_param, 1)
    qc.cx(1, 0)

    for i in range(nb_orbitals - 2):
        qc.cx(i, 2 + i)

    return qc
