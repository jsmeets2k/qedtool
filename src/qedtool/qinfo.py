import numpy as np

from . import relativity


# Pauli matrices
PAULI = np.array([[[1, 0], 
                   [0, 1]],
                  [[0, 1], 
                   [1, 0]],
                  [[0, -1j], 
                   [1j, 0]],
                  [[1, 0], 
                   [0, -1]]])



# Class for quantum states


class QuantumState:
    """
    Class for quantum states.

    Attributes
    ----------
    bra : ndarray
        The bra describing the state.  If the state is mixed, `bra` is `None`.
    ket : ndarray
        The ket describing the state.  If the state is mixed, `ket` is `None`.
    rho : ndarray
        The density matrix describing the state.
    pmu : FourVector or list
        If the QuantumState is a single-particle state with definite 
        4-momentum, then `pmu` is its 4-momentum (instance of FourVector).

    Methods
    -------
    single
        Constructs single-particle polarization states with definite 
        4-momentum.
    mixed
        Generates a mixed quantum state.
    out_state
        Constructs the post-scattering quantum state from the initial state
        and the Feynman amplitudes.
    tensor_product
        Takes the tensor product of two quantum states.
    
    """
    
    def __init__(self):
        
        # Bra and ket
        self.bra = None
        self.ket = None
        
        # Density matrix
        self.rho = None

        # 4-momentum
        self.four_momentum = None
    
    def __add__(self, other): 

        # Check if `other` is a quantum state
        if not isinstance(other, QuantumState):
            return NotImplemented
        
        # Check if the length of the kets are equal
        if len(self.ket) != len(other.ket):
            raise Exception("n-particle states cannot be added to "
                            + "(m != n)-particle states.")

        # The sum of self and other
        sum_bra = self.bra + other.bra
        sum_ket = self.ket + other.ket

        # Create an instance
        sum_state = QuantumState()

        # Assign properties
        sum_state.bra = sum_bra
        sum_state.ket = sum_ket
        sum_state.rho = density_matrix(sum_state)

        # If both have equal `pmu`, their sum has that same `pmu`
        if self.four_momentum == None or other.four_momentum == None:
            sum_state.four_momentum = None
        elif (np.round(self.four_momentum.vector, 10) \
            == np.round(other.four_momentum.vector, 10)).all():
            sum_state.four_momentum = self.four_momentum

        return sum_state
    
    def __sub__(self, other): 

        # Check if `other` is a quantum state
        if not isinstance(other, QuantumState):
            return NotImplemented
        
        # Check if the length of the kets are equal
        if len(self.ket) != len(other.ket):
            raise Exception("n-particle states cannot be added to "
                            + "(m != n)-particle states.")

        # The sum of self and other
        dif_bra = self.bra - other.bra
        dif_ket = self.ket - other.ket

        # Create an instance
        dif_state = QuantumState()

        # Assign properties
        dif_state.bra = dif_bra
        dif_state.ket = dif_ket
        dif_state.rho = density_matrix(dif_state)

        # If both have equal `pmu`, their sum has that same `pmu`
        if self.four_momentum == None or other.four_momentum == None:
            dif_state.four_momentum = None
        elif (np.round(self.four_momentum.vector, 10) \
            == np.round(other.four_momentum.vector, 10)).all():
            dif_state.four_momentum = self.four_momentum

        return dif_state

    def __mul__(self, other):

        # Multiplication with float or int
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, complex) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64) \
                or isinstance(other, np.complex128) \
                or isinstance(other, QuantumState)):
            return NotImplemented
        if not isinstance(self, QuantumState):
            return NotImplemented

        if not isinstance(other, QuantumState):

            # The product of self and other
            scl_bra = self.bra * np.conj(other + 0j)
            scl_ket = self.ket * (other + 0j)

            # Create an instance
            scl_state = QuantumState()

            # Assign properties
            scl_state.bra = scl_bra
            scl_state.ket = scl_ket
            scl_state.rho = density_matrix(scl_state)

            # Assign `four_momentum`
            scl_state.four_momentum = self.four_momentum

            return scl_state
        
        else:

            return tensor_product(self, other)
    
    def __rmul__(self, other):

        # Multiplication with float or int
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, complex) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64) \
                or isinstance(other, np.complex128) \
                or isinstance(other, QuantumState)):
            return NotImplemented
        if not isinstance(self, QuantumState):
            return NotImplemented

        if not isinstance(other, QuantumState):

            # The product of self and other
            scl_bra = self.bra * np.conj(other + 0j)
            scl_ket = self.ket * (other + 0j)

            # Create an instance
            scl_state = QuantumState()

            # Assign properties
            scl_state.bra = scl_bra
            scl_state.ket = scl_ket
            scl_state.rho = density_matrix(scl_state)

            # Assign `pmu`
            scl_state.four_momentum = self.four_momentum

            return scl_state
        
        else:

            return tensor_product(self, other)
    
    def __truediv__(self, other):

        # Multiplication with float or int
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, complex) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64) \
                or isinstance(other, np.complex128)):
            return NotImplemented
        if not isinstance(self, QuantumState):
            return NotImplemented

        # The sum of self and other
        scl_bra = self.bra / np.conj(other + 0j)
        scl_ket = self.ket / (other + 0j)

        # Create an instance
        scl_state = QuantumState()

        # Assign properties
        scl_state.bra = scl_bra
        scl_state.ket = scl_ket
        scl_state.rho = density_matrix(scl_state)

        # Assign `pmu`
        scl_state.four_momentum = self.four_momentum

        return scl_state

    def single(pmu, polarization):
        """
        Return a single-particle polarization eigenstate.

        Parameters
        ----------
        pmu : FourVector
            The definite momentum of the single-particle state.
        polarization : str
            The polarization of the state.  Must be `'L'`, `'R'`, `'H'`, `'V'`,
            `'D'` or `'A'`.

        Returns
        -------
        state : QuantumState
            Single-particle polarization eigenstate.
        
        """
        
        # Make an instance
        state = QuantumState()
        
        # `pmu` must be a FourVector
        if pmu is not None and not isinstance(pmu, relativity.FourVector):
            raise TypeError("`pmu` must be of type `FourVector` or None.")

        # Create single particle kets
        if polarization == 'L':
            state.ket = np.array([1, 0]) 
            state.bra = np.array([1, 0])
        elif polarization == 'R':
            state.ket = np.array([0, 1])
            state.bra = np.array([0, 1])
        elif polarization == 'H':
            state.ket = np.array([1, 1]) / np.sqrt(2)
            state.bra = np.array([1, 1]) / np.sqrt(2)
        elif polarization == 'V':
            state.ket = np.array([1j, -1j]) / np.sqrt(2)
            state.bra = np.array([-1j, 1j]) / np.sqrt(2)
        elif polarization == 'D':
            state.ket = np.array([1 + 1j, 1 - 1j]) / 2
            state.bra = np.array([1 - 1j, 1 + 1j]) / 2
        elif polarization == 'A':
            state.ket = np.array([1 - 1j, 1 + 1j]) / 2
            state.bra = np.array([1 + 1j, 1 - 1j]) / 2
        else:
            raise ValueError("`polarization` must be `'L'`, `'R'`, `'H'`, " \
                             + "`'V'`, `'D'` or `'A'`.")

        # Assign quantum state (rho) property
        state.rho = density_matrix(state)
        
        # Assign 4-momentum (pmu) property
        if pmu == None:
            state.four_momentum = None

        elif pmu is not None:
            state.four_momentum = pmu
        
        return state
    
    def mixed(states, w):
        """
        Construct the in-state, optionally with classical probabilities.

        Parameters
        ----------
        w : list or ndarray
            1D array containing classical probabilities of occupying the
            quantum states in `states`.  If `w` is not normalized, it will be
            automatically normalized to sum(w) = 1.
        states : list or ndarray
            1D array containing the quantum states, instances of type
            `QuantumState`.

        Returns
        -------
        state : QuantumState
            The quantum state of the entering particles.
        
        """
            
        # `states` and `w` checks
        if not isinstance(states, list) and \
            not isinstance(states, np.ndarray):
            raise TypeError("`states` should be a `list` or `ndarray`.")
        if not isinstance(w, list) and \
            not isinstance(w, np.ndarray):
            raise TypeError("`w` should be a `list` or `ndarray`.")

        # Make an instance
        state = QuantumState()

        # Length of w
        length = len(w)

        # Check whether `w` and `states` are of equal size
        if length != len(states):
            raise Exception("`w` and `states` must be of equal size.")
        
        # Dimensionality of Hilbert space
        d = len(states[0].ket)

        # `states` must contain elements of type QuantumState
        for i in range(len(states)):
            if not isinstance(states[i], QuantumState):
                raise Exception("Elements of `states` must be of type \
                                 `QuantumState`.")
            if len(states[i].ket) != d:
                raise Exception("Elements of `states` must all be elements \
                                 of the same Hilbert space.")
            
        # Normalize w
        sum = np.sum(w)
        normalized = w / sum
        
        # Construct the (mixed) density matrix
        rho = np.zeros([d, d])
        for i in range(length):
            rho = rho + normalized[i] * states[i].rho
        
        # Density matrix only, since this might be a mixed state
        state.rho = rho
        
        return state

    def out_state(in_state, amplitudes):
        """
        Construct the out-state from the in-state and Feynman amplitudes in
        the {LL, LR, RL, RR} basis.

        Parameters
        ----------
        in_state : QuantumState
            The in-state with 4x4 density matrix.
        amplitudes : array_like of shape (4, 4)
            An array that contains all Feynman polarized amplitudes.  The first 
            index of `amplitudes` is the final state's helicity configuration, 
            the second index denotes the initial state's helicity
            configuration.

        Notes
        -----
        This density operator is not normalized, such that the differential
        probability is a function of the out-state only, and not Feynman
        amplitudes.
        
        """

        # Make an instance
        state = QuantumState()

        # Check if `in_state` is of type `QuantumState`
        if not isinstance(in_state, QuantumState):
            raise Exception("`in_state` must be of type `QuantumState`.")

        # In-state density matrix
        rho_in = in_state.rho

        # Construct the out-state density matrix from Feynman amplitudes
        rho = np.zeros([4, 4], dtype=np.complex128)
        for a in range(4):
            for b in range(4):
                rho_ab = 0
                for i in range(4):
                    for j in range(4):
                        rho_ab = rho_ab \
                                 + rho_in[i][j] \
                                 * amplitudes[a][i] \
                                 * np.conjugate(amplitudes[b][j])
                rho[a][b] = rho_ab

        # Density matrix only, since this might be a mixed state
        state.rho = rho

        return state



# Other functions for quantum scattering and quantum information quantities


def diff_cross_section(pmu_1, pmu_2, out_state, projection=None):
    """
    Return the 2-to-2 particle differential cross section.

    Parameters
    ----------
    out_state : QuantumState
        The projected out-state.   
    projection : QuantumState
        A quantum state on which the out-state is projected.
    pmu_1, pmu_2 : FourVector
        The initial 4-momenta.

    Returns
    -------
    ds : float
        The differential cross section.

    Notes
    -----
    The `projection` keyword is especially useful when investigating the
    directional emission of some specified quantum state.
        
    """

    # `pmu_1` and `pmu_2` must be of type `FourVector`
    if not isinstance(pmu_1, relativity.FourVector):
        raise TypeError("`pmu_1` must be of type `FourVector`.")
    if not isinstance(pmu_2, relativity.FourVector):
        raise TypeError("`pmu_2` must be of type `FourVector`.")
    
    # Check if `out_state` is a quantum state
    if not isinstance(out_state, QuantumState):
        raise TypeError("`out_state` must be of type `QuantumState`.")
    
    # Check the `projection`
    if projection != None and not isinstance(projection, QuantumState):
        raise TypeError("`projection` must be of type `QuantumState`.")
    
    # Not normalized out-state density matrix
    rho = out_state.rho

    # Energies
    energy_1 = pmu_1.sphericals[0]
    energy_2 = pmu_2.sphericals[0]

    # 3-velocities
    beta_1 = relativity.ThreeVector.beta(pmu_1)
    beta_2 = relativity.ThreeVector.beta(pmu_2)

    # Absolute difference
    diff = beta_1 - beta_2
    abs_diff = np.sqrt(diff * diff)

    # Prefactor
    factor = 1/abs_diff/4/energy_1/energy_2
    
    if projection == None:
        return factor * np.real(np.trace(rho))
    else:
        if rho.shape != projection.rho.shape:
            raise Exception("`out_shape` and `projection` must be quantum "
                            + "states with an equal amount of particles.")
        else:
            return factor * np.real(np.trace(np.dot(rho, projection.rho)))
        

def differential_probability(out_state, projection=None):
    """
    Return the differential scattering probability.

    Parameters
    ----------
    out_state : QuantumState
        The final state.   
    projection : QuantumState
        A quantum state on which the out-state is projected.

    Returns
    -------
    dw : float
        The differential scattering probability.

    Notes
    -----
    The `projection` keyword is especially useful when investigating the
    directional emission of some specified quantum state.
        
    """

    # Check if `out_state` is a quantum state
    if not isinstance(out_state, QuantumState):
        raise TypeError("`out_state` must be of type `QuantumState`.")
    
    if projection != None and not isinstance(projection, QuantumState):
        raise TypeError("`projection` must be of type `QuantumState`.")
    
    # Not normalized out-state density matrix
    rho = out_state.rho
    
    if projection == None:

        return np.real(np.trace(rho))
    
    else:

        if rho.shape != projection.rho.shape:
            raise Exception("`out_shape` and `projection` must be quantum "
                            + "states with an equal amount of particles.")
        else:
            return np.real(np.trace(np.dot(rho, projection.rho)))


def concurrence(state):
    """
    Return the concurrence for a given two-particle state.

    Parameters
    ----------
    state : QuantumState
        The two-particle quantum state of which the concurrence is calculated.

    Returns
    -------
    c : float
        The concurrence of `state`, a real number on the interval [0, 1].
    
    """

    # Check if `state` is of type `QuantumState`
    if not isinstance(state, QuantumState):
        raise Exception("`state` must be of type `QuantumState`.")

    # Density matrix of the state (normalize automatically)
    rho = state.rho

    # Check is `rho` is a 4x4 matrix
    if not rho.shape == (4, 4):
        raise Exception("`state` must be a 2-particle state.")

    # Normalize `rho`
    rho_norm = rho / np.trace(rho)
    
    # Define the spin flip matrix
    spin_flip = np.array([[0, 0, 0, -1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [-1, 0, 0, 0]])
        
    # The Q-matrix (non-Hermitian)
    q = rho_norm.dot(spin_flip).dot(np.conjugate(rho_norm)).dot(spin_flip)
        
    # Find eigenvalues of Q and order them
    eigvals = np.linalg.eig(q)[0]
    descending = np.sort(eigvals)[::-1]
    l1 = descending[0]
    l2 = descending[1]
    l3 = descending[2]
    l4 = descending[3]
        
    # Calculate concurrence
    return np.real(np.max([0, np.sqrt(l1) \
                              - np.sqrt(l2) \
                              - np.sqrt(l3) \
                              - np.sqrt(l4)]))


def stokes_parameter(state, l):
    """
    Return the Stokes parameters of a given quantum state.

    Parameters
    ----------
    state : QuantumState
        The quantum state of which the Stokes parameters are calculated.
    l : ndarray
        1D array containing integers 0, 1, 2 or 3.  These indicate which
        Stokes parameters are to be calculated.

    Returns
    -------
    s : float
        The `l`-th Stokes parameter of the quantum state `state`.
    
    """

    # Check if `state` is of type `QuantumState`
    if not isinstance(state, QuantumState):
        raise Exception("`state` must be of type `QuantumState`.")

    # Normalize the density matrix of `state`
    rho = state.rho / np.trace(state.rho)
        
    # Errors for `l`
    if len(np.array([l])) != 1:
        raise Exception("`l` should be a 1D array.")
    for i in range(len(l)):
        if l[i] not in [0, 1, 2, 3]:
            raise ValueError("Elements of `l` must be either 0, 1, 2 or 3.")
    
    # Matching between `rho` and `l`
    if 2**len(l) != rho.shape[0]:
        raise Exception("`rho` and `l` do not match in dimensions. \
                         If `rho` is an nxn matrix, then 2^len(l) \
                         must equal `n`.")
    
    # Pauli vector
    S0 = PAULI[0]
    
    if l == [0] * len(l):

        for i in range(1, len(l)):
            S0 = np.kron(S0, PAULI[0])
            
        return np.trace(rho.dot(S0))
        
    else:
        
        elems = []
        for i in range(len(l)):
            elems.append(PAULI[l[i]])
            
        ss = elems[0]
        for i in range(1, len(elems)):
            ss = np.kron(ss, elems[i])
            S0 = np.kron(S0, PAULI[0])
            
        s0 = np.trace(rho.dot(S0))

        return np.real(np.trace(rho.dot(ss)) / s0)


def degree_polarization(state):
    """
    Return the 1- or 2-particle degree of polarization.

    Parameters
    ----------
    state : QuantumState
        The 2-particle quantum state of which the degree of polarization is 
        calculated.

    Returns
    -------
    deg_pol : float
        The 2-particle degree of polarization.
    
    """

    # Check if `state` is of type `QuantumState`
    if not isinstance(state, QuantumState):
        raise Exception("`state` must be of type `QuantumState`.")
    
    # Two-particle degree of polarization
    if state.rho.shape == (4, 4):
    
        # Calculate the 0j-Stokes parameters
        s01 = stokes_parameter(state, [0, 1])
        s02 = stokes_parameter(state, [0, 2])
        s03 = stokes_parameter(state, [0, 3])

        # Calculate the j0-Stokes parameters
        s10 = stokes_parameter(state, [1, 0])
        s20 = stokes_parameter(state, [2, 0])
        s30 = stokes_parameter(state, [3, 0])

        return 1 - 0.5 * (s01**2 + s02**2 + s03**2 + s10**2 + s20**2 + s30**2)
    
    # Single-particle degree of polarization
    elif state.rho.shape == (2, 2):

        # Calculate the Stokes parameters
        s1 = stokes_parameter(state, [1])
        s2 = stokes_parameter(state, [2])
        s3 = stokes_parameter(state, [3])
        
        return np.sqrt(s1**2 + s2**2 + s3**2)


def inner_product(state_1, state_2):
    """
    Return the inner product of two quantum states with definite momentum and
    polarization states.

    Parameters
    ----------
    state_1 : QuantumState
        The first quantum state.
    state_2 : QuantumState
        The second quantum state.

    Returns
    -------
    inprod : float or complex
        The inner product of `state_1` and `state_2`.
    
    """

    # Check if `state_1` and `state_2` are of type QuantumState
    if not (isinstance(state_1, QuantumState) \
            or isinstance(state_2, QuantumState)):
        raise Exception("`state_1` and `state_2` must be of type \
                         `QuantumState`.")
    
    # Check if states are pure
    if (state_1.bra == None).any():
        raise Exception("`state_1` must be a pure state.")
    if (state_2.ket == None).any():
        raise Exception("`state_2` must be a pure state.")
        
    # Retreive bra and ket
    bra = state_1.bra
    ket = state_2.ket

    return np.dot(bra, ket)


def tensor_product(state_1, state_2):
        """
        Return the tensor product of two uncorrelated quantum states.

        Parameters
        ----------
        state_1 : QuantumState
            The first quantum state.
        state_2 : Quantum state
            The second quantum state.

        Returns
        -------
        state : QuantumState
            The composite state of `state_1` and `state_2`.
        
        """
        
        # Make an instance
        state = QuantumState()

        # Check if `state_1` and `state_2` are of type QuantumState
        if not (isinstance(state_1, QuantumState) \
                or isinstance(state_2, QuantumState)):
            raise Exception("`state_1` and `state_2` must be of type "
                            + "`QuantumState`.")
        
        # Assign properties
        if np.any(state_1.ket) != None and np.any(state_2.ket) != None:
            state.ket = np.kron(state_1.ket, state_2.ket)
            state.bra = np.conjugate(state.ket)

        state.rho = np.kron(state_1.rho, state_2.rho)
        
        return state


def density_matrix(state):
    """
    Return the density matrix of a quantum state.

    Parameters
    ----------
    state : QuantumState
        The quantum state of which the density matrix will be calculated.
        
    Returns
    -------
    rho : ndarray
        The density matrix representing `state`.

    """

    # Check if `state` is of type `QuantumState`
    if not isinstance(state, QuantumState):
        raise Exception("`state` must be of type `QuantumState`.")
        
    # Obtain bra and ket
    bra = state.bra
    ket = state.ket
        
    return np.outer(ket, bra)

