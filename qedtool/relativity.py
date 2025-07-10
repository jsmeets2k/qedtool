import numpy as np
from copy import deepcopy

from . import qed



# Three- and four-vector classes


class ThreeVector:
    """
    Class for 3-vectors, i.e. the spatial parts of 4-vectors.

    Attributes
    ----------
    vector : ndarray
        The 3-vector as an array.
    cartesians : tuple of shape (3,)
        The Cartesian components of the 3-vector; x, y and z.
    sphericals : tuple of shape (3,)
        The spherical components of the 3-vector; r, theta and phi.
    cylindricals : tuple of shape (3,)
        The cylindrical components of the 3-vector; rho, phi and z.

    Methods
    -------
    dot
        Allows one to multiply a `ThreeVector` with an ndarray of shape (3, 3).
    beta
        Returns the 3-velocity, an instance of `ThreeVector` from a specified
        4-momentum `pmu`, an instance of `FourVector`.
    
    """

    def __init__(self, c1, c2, c3, coordinates='spherical'):
            
        # Construct 3-vector from 4-vector
        vmu = four_vector(0, c1, c2, c3, coordinates)
        self.vector = np.array([vmu[1], vmu[2], vmu[3]])

        # Various components
        if coordinates == 'spherical':
            self.sphericals = (c1, c2, c3)
            self.cylindricals = cylindrical_components(self.vector)
        elif coordinates == 'cylindrical':
            self.sphericals = spherical_components(self.vector)
            self.cylindricals = (c1, c2, c3)
        else:
            self.sphericals = spherical_components(self.vector)
            self.cylindricals = cylindrical_components(self.vector)
        self.cartesians = cartesian_components(self.vector)
    
    def __add__(self, other): 

        # Check if `other` is a 3-vector
        if not isinstance(other, ThreeVector):
            return NotImplemented

        # The sum of self and other
        svec_vector = self.vector + other.vector

        # Cartesian components of the sum
        x, y, z = cartesian_components(svec_vector)

        # Create an instance
        svec = ThreeVector(x, y, z, coordinates='Cartesian')

        return svec
    
    def __sub__(self, other): 

        # Check if `other` is a 3-vector
        if not isinstance(other, ThreeVector):
            return NotImplemented

        # The sum of self and other
        dvec_vector = self.vector - other.vector

        # Cartesian components of the sum
        x, y, z = cartesian_components(dvec_vector)

        # Create an instance
        dvec = ThreeVector(x, y, z, coordinates='Cartesian')

        # The `dvec` as an array
        dvec.vector = dvec_vector

        return dvec

    def __mul__(self, other):

        # Multiplication
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64) \
                or isinstance(other, ThreeVector)):
            return NotImplemented
        if not isinstance(self, ThreeVector):
            return NotImplemented
        
        if not isinstance(other, ThreeVector):
        
            # New 3-vector as array
            scvec = self.vector * other

            # Cartesian components of `scvec`
            x, y, z = cartesian_components(scvec)

            # Make an instance
            scaled_vec = ThreeVector(x, y, z, coordinates='Cartesian')

            # `scaled_vec` as an array
            scaled_vec.vector = scvec

            return scaled_vec
        
        else:

            return euclidean_product(self.vector, other.vector)
    
    def __rmul__(self, other):

        # Multiplication with float or int
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64)):
            return NotImplemented
        if not isinstance(self, ThreeVector):
            return NotImplemented
        
        if not isinstance(other, ThreeVector):
        
            # New 3-vector as array
            scvec = self.vector * other

            # Cartesian components of `scvec`
            x, y, z = cartesian_components(scvec)

            # Make an instance
            scaled_vec = ThreeVector(x, y, z, coordinates='Cartesian')

            # `scaled_vec` as an array
            scaled_vec.vector = scvec

            return scaled_vec
        
        else:

            return euclidean_product(self.vector, other.vector)
    
    def __truediv__(self, other):

        # Multiplication with float or int
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64)):
            return NotImplemented
        if not isinstance(self, ThreeVector):
            return NotImplemented
        
        # New 3-vector as array
        scvec = self.vector / other

        # Cartesian components of `svec`
        x, y, z = cartesian_components(scvec)

        # Make an instance
        scaled_vec = ThreeVector(x, y, z, coordinates='Cartesian')

        # `vmu` as a vector
        scaled_vec.vector = scvec

        return scaled_vec

    def dot(mat, vec):
        """
        Return the matrix product of a 3x3 matrix and a 3-vector.

        Parameters
        ----------
        mat : ndarray of shape (3, 3)
            The matrix that will be multiplied with the 3-vector.
        vec : ThreeVector
            The 3-vector that is multiplied with `mat`.

        Returns
        -------
        mv : ThreeVector
            The matrix product of `mat` and `vec`.
        
        """

        # Make sure that `mat` is an ndarray
        matrix = np.array(mat)

        # Check if `vector` is of instance `ThreeVector`
        if not isinstance(vec, ThreeVector):
            raise TypeError("`vector` must be an instance `ThreeVector`.")

        # Check if `matrix` is a 3x3 matrix
        if matrix.shape != (3, 3):
            raise Exception("`matrix` must be an ndarray of shape (3, 3).")

        # Make an instance
        mv = ThreeVector(1, 0, 0)

        # `mv` as an ndarray
        v = np.dot(matrix, vec.vector)
        mv.vector = np.array([v[0], v[1], v[2]])
        
        # Overwrite components
        mv.sphericals = spherical_components(mv.vector)
        mv.cartesians = cartesian_components(mv.vector)
        mv.cylindricals = cylindrical_components(mv.vector)

        return mv

    def beta(pmu):
        """
        Return the 3-velocity of a 4-momentum.

        Parameters
        ----------
        pmu : FourVector
            The 4-vector of which the 3-velocity will be calculated.

        Returns
        -------
        beta : ThreeVector
            The 3-velocity, sometimes also referred to as the boost vector.
        
        """

        # Check if `pmu` is of type `FourVector`
        if isinstance(pmu, FourVector):
            theta, phi = spherical_components(pmu.vector)[2:4]
        else:
            raise Exception("`pmu` must be of type `FourVector`.")
        
        # Calculate the norm of the 3-velocity
        b = np.sqrt(euclidean_product(three_velocity(pmu.vector), 
                                      three_velocity(pmu.vector)))

        # Make an instance
        beta = ThreeVector(b, theta, phi)

        # The vector as an array
        beta.vector = three_velocity(pmu.vector)

        return beta


class FourVector:
    """
    Class for 4-vectors, i.e. rank-1 Lorentz tensors.

    Attributes
    ----------
    vector : ndarray of shape (4,)
        The 4-vector expressed as an array.
    cartesians : tuple of shape (3,)
        The Cartesian x-, y- and z-components of the 4-vector.
    sphericals : tuple of shape (3,)
        The spherical r-, theta- and phi-components of the 4-vector.
    cylindricals : tuple of shape (3,)
        The spherical rho-, phi- and z-components of the 4-vector.

    Methods
    -------
    dot
        Allows the user to multiply instances of `FourVector` with ndarrays
        of shape (4, 4).
    polarization
        Constructs the polarization of a photon, from a specified helicity
        and 4-momentum.
    dirac_current
        Constructs the Dirac current from two Dirac spinors.
    
    """

    def __init__(self, c0, c1, c2, c3, coordinates='spherical'):
        
        # The 4-vector as array
        self.vector = four_vector(c0, c1, c2, c3, coordinates)

        if np.imag(c0) == 0.0 and np.imag(c1) == 0.0 and np.imag(c2) == 0.0 \
        and np.imag(c3) == 0.0:

            # Various components
            if coordinates == 'spherical':
                self.sphericals = (c0, c1, c2, c3)
                self.cylindricals = cylindrical_components(self.vector)
            elif coordinates == 'cylindrical':
                self.sphericals = spherical_components(self.vector)
                self.cylindricals = (c0, c1, c2, c3)
            elif coordinates == 'Cartesian':
                self.sphericals = spherical_components(self.vector)
                self.cylindricals = cylindrical_components(self.vector)

        else:

            self.sphericals = None
            self.cylindricals = None

        self.cartesians = cartesian_components(self.vector)

    def __add__(self, other): 

        # Check if `other` is a 4-vector
        if not isinstance(other, FourVector):
            return NotImplemented

        # The sum of self and other
        smu_vector = self.vector + other.vector

        # Cartesian components of the sum
        t, x, y, z = cartesian_components(smu_vector)

        # Create an instance
        smu = FourVector(t, x, y, z, coordinates='Cartesian')

        # The 4-vector as array
        smu.vector = smu_vector

        return smu
    
    def __sub__(self, other): 

        # Check if `other` is a 4-vector
        if not isinstance(other, FourVector):
            return NotImplemented

        # The sum of self and other
        dmu_vector = self.vector - other.vector

        # Cartesian components of the sum
        t, x, y, z = cartesian_components(dmu_vector)

        # Create an instance
        dmu = FourVector(t, x, y, z, coordinates='Cartesian')

        # The 4-vector as array
        dmu.vector = dmu_vector

        return dmu

    def __mul__(self, other):

        # Multiplication with float or int
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, complex) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64) \
                or isinstance(other, np.complex128) \
                or isinstance(other, FourVector)):
            return NotImplemented
        if not isinstance(self, FourVector):
            return NotImplemented
        
        if not isinstance(other, FourVector):
        
            # New 4-vector as array
            svmu = self.vector * other

            # Cartesian components of `svmu`
            t, x, y, z = cartesian_components(svmu)

            # Make an instance
            scaled_vmu = FourVector(t, x, y, z, coordinates='Cartesian')

            # `scaled_vmu` as a vector
            scaled_vmu.vector = svmu

            return scaled_vmu
        
        else:

            # Return the Lorentzian inner product
            return lorentzian_product(self.vector, other.vector)
    
    def __rmul__(self, other):

        # Multiplication with float or int
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, complex) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64) \
                or isinstance(other, np.complex128) \
                or isinstance(other, FourVector)):
            return NotImplemented
        if not isinstance(self, FourVector):
            return NotImplemented
        
        if not isinstance(other, FourVector):
        
            # New 4-vector as array
            svmu = self.vector * other

            # Cartesian components of `svmu`
            t, x, y, z = cartesian_components(svmu)

            # Make an instance
            scaled_vmu = FourVector(t, x, y, z, coordinates='Cartesian')

            # `scaled_vmu` as a vector
            scaled_vmu.vector = svmu

            return scaled_vmu
        
        else:

            # Return the Lorentzian inner product
            return lorentzian_product(self.vector, other.vector)
    
    def __truediv__(self, other):

        # Multiplication with float or int
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, complex) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64) \
                or isinstance(other, np.complex128)):
            return NotImplemented
        if not isinstance(self, FourVector):
            return NotImplemented
        
        # New 4-vector as array
        svmu = self.vector / other

        # Spherical components of `svmu`
        t, x, y, z = cartesian_components(svmu)

        # Make an instance
        scaled_vmu = FourVector(t, x, y, z, coordinates='Cartesian')

        # `scaled_vmu` as a vector
        scaled_vmu.vector = svmu

        return scaled_vmu

    def dot(mat, vmu):
        """
        Return the matrix product of a 4x4 matrix and a 4-vector.

        Parameters
        ----------
        matrix : ndarray of shape (4, 4)
            The matrix that will be multiplied with the 4-vector.
        vmu : FourVector
            The 4-vector that is multiplied with `matrix`.

        Returns
        -------
        mvmu : ThreeVector
            The matrix product of `matrix` and `vmu`.
        
        """

        # Make sure that `mat` is an ndarray
        matrix = np.array(mat)

        # Check if `vector` is of instance `FourVector`
        if not isinstance(vmu, FourVector):
            raise TypeError("`vector` must be an instance `FourVector`.")

        # Check if `matrix` is a 4x4 matrix
        if matrix.shape != (4, 4):
            raise Exception("`matrix` must be an ndarray of shape (4, 4).")

        # Make an instance
        mvmu = FourVector(1, 1, 0, 0)

        # `mat_vec` as an ndarray
        mv = np.dot(matrix, vmu.vector)
        mvmu.vector = four_vector(mv[0], mv[1], mv[2], mv[3], 
                                     coordinates='Cartesian')
        
        # Overwrite components
        mvmu.sphericals = spherical_components(mvmu.vector)
        mvmu.cartesians = cartesian_components(mvmu.vector)
        mvmu.cylindricals = cylindrical_components(mvmu.vector)

        return mvmu

    def polarization(helicity, pmu, conjugate=False):
        """
        Return the 4-polarization (of a photon).

        Parameters
        ----------
        helicity : int
            The helicity of the photon.  Must be +1 or -1 for photons.
        pmu : FourVector
            The 4-momentum of the photon.
        conjugate : bool
            If `conjugate` is `True` then the complex conjugate of the 
            4-polarization is taken.  It equals `False` by default.

        Returns
        -------
        emu : FourVector
            The 4-polarization.

        """

        # Check whether `pmu` is a 4-vector
        if not isinstance(pmu, FourVector):
            raise TypeError("`pmu` must be of type `FourVector`.")
        
        # Make an instance, but the coordinates are to be overwritten
        emu = FourVector(0, 1, 0, 0, coordinates='spherical')

        # The vector as an array
        if conjugate == False:
            emu.vector = qed.photon_polarization(helicity, pmu.vector)
        elif conjugate == True:
            emu.vector = np.conjugate(qed.photon_polarization(helicity, 
                                                              pmu.vector))
        else:
            raise Exception("`conjugate` must be `True` of `False`.")

        # Overwrite all attributes
        emu.sphericals = None
        emu.cartesians = cartesian_components(emu.vector)
        emu.cylindricals = None

        return emu

    def dirac_current(psi_1, psi_2):
        """
        Return the Dirac current from two Dirac spinors.

        Parameters
        ----------
        psi_1 : DiracSpinor
            The adjointed Dirac spinor.
        psi_2 : DiracSpinor
            The second Dirac spinor.

        Returns
        -------
        jmu : FourVector
            The Dirac current; a 4-vector.
        
        """

        # Check whether `psi_1` and `psi_2` are of type `DiracSpinor`
        if not (isinstance(psi_1, qed.DiracSpinor) \
                or isinstance(psi_1, qed.RealParticle)):
            raise Exception("`psi_1` must be of type `DiracSpinor` "\
                            + "or `RealParticle`.")
        if not (isinstance(psi_2, qed.DiracSpinor) \
                or isinstance(psi_2, qed.RealParticle)):
            raise Exception("`psi_2` must be of type `DiracSpinor` "\
                            + "or `RealParticle`.")
        
        # Check whether `psi_1` is adjointed
        if psi_1.adjoint == False:
            raise Exception("`psi_1` must be adjointed.")
        elif psi_2.adjoint == True:
            raise Exception("`psi_2` should not be adjointed.")
        
        if isinstance(psi_1, qed.DiracSpinor) \
        and isinstance(psi_2, qed.DiracSpinor):
        
            # Construct the Dirac current
            current = qed.dirac_current(psi_1.bispinor, psi_2.bispinor)

        elif isinstance(psi_1, qed.RealParticle) \
        and isinstance(psi_2, qed.RealParticle):
            
            # Construct the Dirac current
            current = qed.dirac_current(psi_1.polarization.bispinor, 
                                        psi_2.polarization.bispinor)
            
        else:

            # `psi_1` and `psi_2` msut be of the same type of course
            raise Exception("`psi_1` and `psi_2` must be of the same type.")
        
        # Make an instance
        jmu = FourVector(0, 1, 0, 0)

        # The vector as an array
        jmu.vector = current

        # Overwrite all attributes here
        jmu.sphericals = spherical_components(jmu.vector)
        jmu.cartesians = cartesian_components(jmu.vector)
        jmu.cylindricals = cylindrical_components(jmu.vector)

        return jmu



# Functions to construct three- and four-vectors


def four_vector(c0, c1, c2, c3, coordinates='spherical'):
    """
    Return a 4-vector.
    
    Parameters
    ----------
    c0, c1, c2, c3 : float
        The time-component `c0` and spatial coordinates `c1`, `c2` and `c3`
        of the 4-vector.  The meaning of the spatial components depends on the
        `coordinates` keyword.
    coordinates : str
        The coordinate system.  If `coordinates` equals "spherical", then
        `c1`, `c2` and `c3` signify the radial, polar and azimuthal components
        of the 4-vector.  When `coordinates` equals "Cartesian", they signify
        the x-, y- and z-components.  If it is "cylindrical", then they refer
        to the cylindrical components rho, phi and z.
        
    Returns
    -------
    vmu : ndarray of shape (4,)
        Generated 4-vector.
    
    """

    if coordinates == 'spherical':
    
        return np.array([c0,
                         c1 * np.sin(c2) * np.cos(c3),
                         c1 * np.sin(c2) * np.sin(c3),
                         c1 * np.cos(c2)])
    
    elif coordinates == 'Cartesian':
    
        return np.array([c0, c1, c2, c3])
    
    elif coordinates == 'cylindrical':
    
        return np.array([c0, c1 * np.cos(c2), c1 * np.sin(c2), c3])


def three_velocity(pmu):
    """
    Return the 3-velocity of an on-shell 4-momentum.
    
    Parameters
    ----------
    pmu : ndarray of shape (4,) or FourVector
        On-shell 4-momentum.
    
    Returns
    -------
    beta : ndarray of shape (3,)
        3-velocity corresponding to `pmu`.
    
    """

    # Also include the possibility for `pmu` to be a `FourVector`
    if isinstance(pmu, np.ndarray):
        pass
    elif isinstance(pmu, FourVector):
        pmu = pmu.vector

    p0 = pmu[0]
    px = pmu[1]
    py = pmu[2]
    pz = pmu[3]

    p = np.sqrt(px**2 + py**2 + pz**2)

    # Avoid problems when `p0` and `p` are close in value
    if np.round(p0, 10) < np.round(p, 10):
        raise Exception("`pmu` must be light-like or time-like.")
    else:
        m = np.sqrt(np.round(p0, 10)**2 - np.round(p, 10)**2)
    
    n = np.array([px, py, pz]) / p
    beta = np.sqrt(1 - m**2 / p0**2)
    
    return beta * n



# Functions related to boosts


def lorentz_factor(pmu):
    """
    Return the Lorentz factor from an on-shell 4-momentum.
    
    Parameters
    ----------
    pmu : ndarray of shape (4,) or FourVector
        On-shell 4-momentum of which the Lorentz factor is calculated.
    
    Returns
    -------
    gamma : float
        Lorentz factor of the boost.  A real number on the interval [1, +inf).
    
    """

    # Also include the possibility for `pmu` to be a `FourVector`
    if isinstance(pmu, np.ndarray):
        pass
    elif isinstance(pmu, FourVector):
        pmu = pmu.vector

    p0 = pmu[0]
    px = pmu[1]
    py = pmu[2]
    pz = pmu[3]

    p = np.sqrt(px**2 + py**2 + pz**2)

    if p0 == p:
        raise Exception("`pmu` must be time-like and on-shell.")

    velocity = three_velocity(pmu)

    bx = velocity[0]
    by = velocity[1]
    bz = velocity[2]

    b2 = bx**2 + by**2 + bz**2

    return 1/np.sqrt(1 - b2)


def rapidity(pmu):
    """
    Return the rapidity of an on-shell 4-momentum.
    
    Parameters
    ----------
    pmu : ndarray of shape (4,) or FourVector
        On-shell 4-momentum of which the rapidity is calculated.
    
    Returns
    -------
    eta : float
        Rapidity of `pmu`.  A real number on the interval (-inf, +inf).
    
    """

    # Also include the possibility for `pmu` to be a `FourVector`
    if isinstance(pmu, np.ndarray):
        pass
    elif isinstance(pmu, FourVector):
        pmu = pmu.vector

    gamma = lorentz_factor(pmu)
    
    return np.arccosh(gamma)


def pseudorapidity(pmu, n=None):
    """
    Return the pseudorapidity of an on-shell 4-momentum with respect to a
    specified axis.
    
    Parameters
    ----------
    pmu : ndarray of shape (4,) or FourVector
        On-shell 4-momentum of which the pseudorapidity is calculated.
    n : ndarray of shape (3,) or ThreeVector, optional
        Axis to which the pseudorapidity is calculated.  Not necessarily
        normalized to unity.  Default is the z-axis.
    
    Returns
    -------
    y : float
        Pseudorapidity of `pmu`.  A real number on the interval [0, inf).
    
    """

    # Also include the possibility for `pmu` to be a `FourVector`
    if isinstance(pmu, np.ndarray):
        pass
    elif isinstance(pmu, FourVector):
        pmu = pmu.vector

    # Also include the possibility for `n` to be a `ThreeVector`
    if isinstance(n, np.ndarray):
        pass
    elif isinstance(n, ThreeVector):
        n = n.vector

    if n == None:
        n = np.array([0, 0, 1])
    
    if n[0]**2 + n[1]**2 + n[2]**2 != 1:
        n = n / np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    
    p = np.array([pmu[1], pmu[2], pmu[3]])
    pn = np.dot(p, n)
    theta = np.arccos(pn/np.sqrt(p[0]**2 + p[1]**2 + p[2]**2))
    
    return -np.log(np.tan(theta/2))


def boost_matrix(beta, representation):
    """
    Return the boost matrix of a representation.

    Parameters
    ----------
    beta : ThreeVector or ndarray of shape (3,)
        The boost vector.
    representation : str
        The representation, which must be '4-vector' or 'Dirac spinor'.
        Optionally, 'four-vector' or 'bispinor'.

    Returns
    -------
    boost_matrix : ndarray of shape (4, 4)
        The boost matrix of the specified representation.
    
    """

    # Also include the possibility for `beta` to be a `ThreeVector`
    if isinstance(beta, ThreeVector):
        beta = beta.vector
    elif isinstance(beta, np.ndarray):
        if len(beta) == 3:
            pass
    else:
        raise TypeError("`beta` must be of type `ThreeVector` or an " \
                        + "ndarray of shape (3,).")

    if representation == '4-vector' or representation == 'four-vector':

        # Boost vector components
        beta_x = beta[0]
        beta_y = beta[1]
        beta_z = beta[2]
        b = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
        gamma = 1/np.sqrt(1-b**2)

        # Matrix
        boost_matrix = np.array([
                        
                    [gamma, 
                    - gamma * beta_x, 
                    - gamma * beta_y, 
                    - gamma * beta_z],
                        
                    [- gamma * beta_x, 
                    1 + gamma**2 * beta_x**2 / (1 + gamma), 
                    gamma**2 * beta_x * beta_y / (1 + gamma), 
                    gamma**2 * beta_x * beta_z / (1 + gamma)],
                        
                    [- gamma * beta_y, 
                    gamma**2 * beta_x * beta_y / (1 + gamma),
                    1 + gamma**2 * beta_y**2 / (1 + gamma),
                    gamma**2 * beta_y * beta_z / (1 + gamma)],
                        
                    [- gamma * beta_z,
                    gamma**2 * beta_x * beta_z / (1 + gamma),
                    gamma**2 * beta_y * beta_z / (1 + gamma),
                    1 + gamma**2 * beta_z**2 / (1 + gamma)]
                    
                ])
        
    elif representation == 'Dirac spinor' or representation == 'bispinor':

        # Unit vector
        b = np.sqrt(euclidean_product(beta, beta))
        n = beta / b
        nx = n[0]
        ny = n[1]
        nz = n[2]
        
        # Rapidity
        eta = np.arctanh(b)
        
        # Boost matrix
        ns = -2 * (nx * qed.S01 + ny * qed.S02 + nz * qed.S03) / 1j
        boost_matrix = qed.I4 * np.cosh(eta / 2) - ns * np.sinh(eta / 2)

    return boost_matrix


def boost(obj, beta):
    """
    Boost an object.
    
    Parameters
    ----------
    obj : FourVector, DiracSpinor, RealParticle or VirtualParticle
        Object on which the boost is performed.
    beta : ThreeVector
        Boost vector of the boost.  It specifies the velocity of the boost.
        
    Returns
    -------
    obj2 : FourVector, DiracSpinor, RealParticle or VirtualParticle
        Boosted object.
    
    """

    def vector_boost(vmu, beta):

        # Also include the possibility for `vmu` to be a `FourVector`
        if isinstance(vmu, np.ndarray):
            pass
        elif isinstance(vmu, FourVector):
            vmu = vmu.vector

        # Also include the possibility for `beta` to be a `ThreeVector`
        if isinstance(beta, np.ndarray):
            pass
        elif isinstance(beta, ThreeVector):
            beta = beta.vector

        # Obtain the boost matrix
        matrix = boost_matrix(beta, 'four-vector')
            
        return np.array(np.dot(matrix, vmu))
    
    def spinor_boost(psi, beta):

        # Also include the possibility for `psi` to be a `DiracSpinor`
        if isinstance(psi, qed.DiracSpinor):
            adjoint = psi.adjoint
            psi = psi.bispinor
        elif isinstance(psi, np.ndarray):
            if len(psi) == 4:
                pass
        else:
            raise TypeError("`psi` must be of type `DiracSpinor` or an " \
                            + "ndarray of shape (4,).")

        # Also include the possibility for `beta` to be a `ThreeVector`
        if isinstance(beta, ThreeVector):
            beta = beta.vector
        elif isinstance(beta, np.ndarray):
            if len(beta) == 3:
                pass
        else:
            raise TypeError("`beta` must be of type `ThreeVector` or an " \
                            + "ndarray of shape (3,).")
        
        # Obtain the boost matrix
        matrix = boost_matrix(beta, 'Dirac spinor')
        
        # Boost depending on whether `adjoint` is True or False
        if adjoint == False:
            return matrix.dot(psi)
        elif adjoint == True:
            return (psi).dot(np.linalg.inv(matrix))
        else:
            raise Exception("The `adjoint` attribute of a `DiracSpinor` must be " \
                            + "`True` or `False`.")

    # Check if `beta` is a three-vector
    if not isinstance(beta, ThreeVector):
        raise TypeError("`beta` must be of type `ThreeVector`.")

    # If the object is a four-vector
    if isinstance(obj, FourVector):
        vmu_b = vector_boost(obj, beta)
        return FourVector(vmu_b[0], vmu_b[1], vmu_b[2], vmu_b[3], 'Cartesian')
    
    # If the object is a Dirac spinor
    elif isinstance(obj, qed.DiracSpinor):
        psi_b = spinor_boost(obj, beta)
        obj2 = deepcopy(obj)
        obj2.bispinor = np.array([psi_b[0], psi_b[1], psi_b[2], psi_b[3]])
        return obj2
    
    # If the object is a real particle
    elif isinstance(obj, qed.RealParticle):

        # Copy `obj`
        obj2 = deepcopy(obj)

        # Boost the momentum
        pmu_b = vector_boost(obj.pmu, beta)
        obj2.pmu = FourVector(pmu_b[0], pmu_b[1], 
                              pmu_b[2], pmu_b[3], 'Cartesian')

        # If `obj2` is a fermion, boost the fermion representation
        if obj2.species == 'electron' or obj2.species == 'positron' or \
           obj2.species == 'muon' or obj2.species == 'antimuon':
            
            # Boosted polarization
            psi_b = spinor_boost(obj.polarization, beta)
            obj2.polarization.bispinor = np.array([psi_b[0], psi_b[1], \
                                                   psi_b[2], psi_b[3]])
        
        # If `obj2` is a photon, transform in the vector representation
        elif obj2.species == 'photon':

            # Boosted polarization
            emu_b = vector_boost(obj.polarization)
            obj2.polarization.vector = np.array([emu_b[0], emu_b[1], \
                                                 emu_b[2], emu_b[3]])

        return obj2
    
    # If the object is a virtual particle
    elif isinstance(obj, qed.VirtualParticle):

        # Copy `obj`
        obj2 = deepcopy(obj)

        # Boost the momentum
        obj2.pmu = vector_boost(obj.pmu, beta)

        # If `obj2` is a fermion, boost in the double fermion representation
        if obj2.species == 'electron' or obj2.species == 'positron' or \
           obj2.species == 'muon' or obj2.species == 'antimuon' or \
           obj2.species == 'tau' or obj2.species == 'antitau':
            
            # Convert to ndarray
            beta = beta.vector

            # Unit vetcor
            b = np.sqrt(lorentzian_product(beta, beta))
            n = beta / b
            nx = n[0]
            ny = n[1]
            nz = n[2]
            
            # Rapidity
            eta = np.arctanh(b)
            
            # Boost matrix
            ns = -2 * (nx * qed.S01 + ny * qed.S02 + nz * qed.S03) / 1j
            boost = qed.I4 * np.cosh(eta / 2) - ns * np.sinh(eta / 2)
            boost_inv = np.linalg.inv(boost)
            
            obj2.propagator = boost.dot(obj.propagator).dot(boost_inv)
        
        # If `obj` is a photon, transform in the double vector representation
        elif obj2.species == 'photon':
            pass

        return obj2



# Functions related to rotations


def rotation_matrix(angle_vec, representation):
    """
    Return the rotation matrix of a representation.

    Parameters
    ----------
    angle_vec : ThreeVector or ndarray of shape (3,)
        The angle vector.
    representation : str
        The representation, which must be '3-vector', '4-vector' or 
        'Dirac spinor'. Optionally, 'three-vector', 'four-vector' or
        'bispinor'.

    Returns
    -------
    rotation_matrix : ndarray of shape (3, 3) or (4, 4)
        The rotation matrix of the specified representation.
    
    """

    if isinstance(angle_vec, ThreeVector):
        angle_vec = angle_vec.vector
    elif isinstance(angle_vec, np.ndarray):
        if len(angle_vec) == 3:
            pass
    else:
        raise TypeError("`angle_vec` must be of type `ThreeVector` or an " \
                        + "ndarray of shape (3,).")
    
    # Angle vector components
    theta_x = angle_vec[0]
    theta_y = angle_vec[1]
    theta_z = angle_vec[2]
    theta = np.sqrt(theta_x**2 + theta_y**2 + theta_z**2)

    # Unit vector
    n = angle_vec / theta
    nx = n[0]
    ny = n[1]
    nz = n[2]

    if representation == '3-vector' or representation == 'three-vector':

        # Matrix
        rotation_matrix = np.array([

            [nx**2 * (1 - np.cos(theta)) + np.cos(theta), 
             nx * ny * (1 - np.cos(theta)) - nz * np.sin(theta), 
             nx * nz * (1 - np.cos(theta)) + ny * np.sin(theta)],

            [nx * ny * (1 - np.cos(theta)) + nz * np.sin(theta), 
             ny**2 * (1 - np.cos(theta)) + np.cos(theta), 
             ny * nz * (1 - np.cos(theta)) - nx * np.sin(theta)],
            
            [nx * nz * (1 - np.cos(theta)) - ny * np.sin(theta), 
             ny * nz * (1 - np.cos(theta)) + nx * np.sin(theta), 
             nz**2 * (1 - np.cos(theta)) + np.cos(theta)]
             
        ])
    
    elif representation == '4-vector' or representation == 'four-vector':

        # Matrix
        rotation_matrix = np.array([

            [0, 0, 0, 0],

            [0, 
             nx**2 * (1 - np.cos(theta)) + np.cos(theta), 
             nx * ny * (1 - np.cos(theta)) - nz * np.sin(theta), 
             nx * nz * (1 - np.cos(theta)) + ny * np.sin(theta)],

            [0, 
             nx * ny * (1 - np.cos(theta)) + nz * np.sin(theta), 
             ny**2 * (1 - np.cos(theta)) + np.cos(theta), 
             ny * nz * (1 - np.cos(theta)) - nx * np.sin(theta)],
            
            [0, 
             nx * nz * (1 - np.cos(theta)) - ny * np.sin(theta), 
             ny * nz * (1 - np.cos(theta)) + nx * np.sin(theta), 
             nz**2 * (1 - np.cos(theta)) + np.cos(theta)]
             
        ])
        
    elif representation == 'Dirac spinor' or representation == 'bispinor':
        
        # Boost matrix
        ns = 2 * (nx * qed.S23 + ny * qed.S13 + nz * qed.S12)
        rotation_matrix = qed.I4 * np.cos(theta / 2) \
                          - 1j * ns * np.sin(theta / 2)

    return rotation_matrix


def rotation(obj, angle_vec):
    """
    Rotate an object.
    
    Parameters
    ----------
    obj : ThreeVector, FourVector, DiracSpinor, RealParticle or VirtualParticle
        Object on which the rotation is performed.
    angle_vec : ndarray of shape (3,) or ThreeVector
        Angle vector of the rotation.  Its magnitude is the angle of rotation
        and the direction is the axis of rotation.
        
    Returns
    -------
    obj2 : ThreeVector, FourVector, DiracSpinor, RealParticle or 
           VirtualParticle
        Rotated object.
    
    """

    def vector_rotation(vec, angle_vec):

        if isinstance(angle_vec, np.ndarray):
            if len(angle_vec) != 3:
                raise Exception("If `angle_vec` is an ndarray, it must have " \
                                + "shape (3,).")
        elif isinstance(angle_vec, ThreeVector):
            angle_vec = angle_vec.vector
        
        if isinstance(vec, np.ndarray):
            if len(vec) == 3:
                rotation = rotation_matrix(angle_vec, 'three-vector')
            elif len(vec) == 4:
                rotation = rotation_matrix(angle_vec, 'four-vector')
        elif isinstance(vec, FourVector):
            vec = vec.vector
            rotation = rotation_matrix(angle_vec, 'four-vector')
        elif isinstance(vec, ThreeVector):
            vec = vec.vector
            rotation = rotation_matrix(angle_vec, 'three-vector')
            
        return np.array(np.dot(rotation, vec))

    def spinor_rotation(psi, angle_vec):

        if isinstance(psi, qed.DiracSpinor):
            adjoint = psi.adjoint
            psi = psi.bispinor
        elif isinstance(psi, np.ndarray):
            if len(psi) == 4:
                pass
        else:
            raise TypeError("`psi` must be of type `DiracSpinor` or an " \
                            + "ndarray of shape (4,).")

        if isinstance(angle_vec, ThreeVector):
            angle_vec = angle_vec.vector
        elif isinstance(angle_vec, np.ndarray):
            if len(angle_vec) == 3:
                pass
        else:
            raise TypeError("`angle_vec` must be of type `ThreeVector` or an " \
                            + "ndarray of shape (3,).")
        
        # Obtain the boost matrix
        rotation = rotation_matrix(angle_vec, 'Dirac spinor')
        
        # Rotating depending on whether `adjoint` is True or False
        if adjoint == False:
            return rotation.dot(psi)
        elif adjoint == True:
            return (psi).dot(np.linalg.inv(rotation))
        else:
            raise Exception("The `adjoint` attribute of a `DiracSpinor` must be " \
                            + "`True` or `False`.")

    # Check if `angle_vec` is a three-vector
    if not isinstance(angle_vec, ThreeVector):
        raise TypeError("`angle_vec` must be of type `ThreeVector`.")

    # If the object is a three-vector
    if isinstance(obj, ThreeVector):
        v_r = vector_rotation(obj, angle_vec)
        return ThreeVector(v_r[0], v_r[1], v_r[2], 'Cartesian')
    
    # If the object is a four-vector
    if isinstance(obj, FourVector):
        vmu_r = vector_rotation(obj, angle_vec)
        return FourVector(vmu_r[0], vmu_r[1], vmu_r[2], vmu_r[3], 'Cartesian')
    
    # If the object is a Dirac spinor
    elif isinstance(obj, qed.DiracSpinor):
        psi_r = spinor_rotation(obj, angle_vec)
        obj2 = deepcopy(obj)
        obj2.bispinor = np.array([psi_r[0], psi_r[1], psi_r[2], psi_r[3]])
        return obj2
    
    # If the object is a real particle
    elif isinstance(obj, qed.RealParticle):

        # Copy `obj`
        obj2 = deepcopy(obj)

        # Rotate the momentum
        pmu_r = vector_rotation(obj.pmu, angle_vec)
        obj2.pmu = FourVector(pmu_r[0], pmu_r[1], 
                              pmu_r[2], pmu_r[3], 'Cartesian')

        # If `obj2` is a fermion, rotate in the fermion representation
        if obj2.species == 'electron' or obj2.species == 'positron' or \
           obj2.species == 'muon' or obj2.species == 'antimuon' or \
           obj2.species == 'tau' or obj2.species == 'antitau':
            
            # Boosted polarization
            psi_r = spinor_rotation(obj.polarization, angle_vec)
            obj2.polarization.bispinor = np.array([psi_r[0], psi_r[1], \
                                                   psi_r[2], psi_r[3]])
        
        # If `obj2` is a photon, rotate in the vector representation
        elif obj2.species == 'photon':

            # Boosted polarization
            emu_r = vector_rotation(obj.polarization)
            obj2.polarization.vector = np.array([emu_r[0], emu_r[1], \
                                                 emu_r[2], emu_r[3]])

        return obj2
    
    # If the object is a virtual particle
    elif isinstance(obj, qed.VirtualParticle):

        # Copy `obj`
        obj2 = deepcopy(obj)

        # Boost the momentum
        obj2.pmu = vector_rotation(obj.pmu, angle_vec)

        # If `obj2` is a fermion, boost in the double fermion representation
        if obj2.species == 'electron' or obj2.species == 'positron' or \
           obj2.species == 'muon' or obj2.species == 'antimuon' or \
           obj2.species == 'tau' or obj2.species == 'antitau':
            
            # Convert to ndarray
            beta = beta.vector

            # Unit vetcor
            b = np.sqrt(lorentzian_product(beta, beta))
            n = beta / b
            nx = n[0]
            ny = n[1]
            nz = n[2]
            
            # Rapidity
            eta = np.arctanh(b)
            
            # Boost matrix
            ns = -2 * (nx * qed.S01 + ny * qed.S02 + nz * qed.S03) / 1j
            boost = qed.I4 * np.cosh(eta / 2) - ns * np.sinh(eta / 2)
            boost_inv = np.linalg.inv(boost)
            
            obj2.propagator = boost.dot(obj.propagator).dot(boost_inv)
        
        # If `obj` is a photon, transform in the double vector representation
        elif obj2.species == 'photon':
            pass

        return obj2



# Vector components and products


def lorentzian_product(vmu_1, vmu_2):
    """
    Return the Lorentzian product in the mostly minus convention. 
    
    Parameters
    ----------
    vmu_1 : ndarray of shape (4,) or FourVector
        First 4-vector.
    vmu_2 : ndarray of shape (4,) or FourVector
        Second 4-vector.
        
    Returns
    -------
    norm : float
        Lorentzian product of `vmu_1` and `vmu_2`.
    
    """

    # Also include the possibility for `vmu_1` to be a `FourVector`
    if isinstance(vmu_1, np.ndarray):
        pass
    elif isinstance(vmu_1, FourVector):
        vmu_1 = vmu_1.vector

    # Also include the possibility for `vmu_2` to be a `FourVector`
    if isinstance(vmu_2, np.ndarray):
        pass
    elif isinstance(vmu_2, FourVector):
        vmu_2 = vmu_2.vector

    # Errors
    if len(vmu_1) != 4 or len(vmu_2) != 4:
        raise Exception("vmu_1 and vmu_2 must have shape (4,).")
    
    return vmu_1[0] * vmu_2[0] \
           - vmu_1[1] * vmu_2[1] \
           - vmu_1[2] * vmu_2[2] \
           - vmu_1[3] * vmu_2[3]


def euclidean_product(vec_1, vec_2):
    """
    Return the Euclidean product of two 3-vectors.

    Parameters
    ----------
    vec_1 : ndarray of shape (3,)
        The first 3-vector.
    vec_2 : ndarray of shape (3,)
        The second 3-vector.

    Returns
    -------
    prod : float
        Euclidean product of `vec_1` with `vec_2`.

    """

    return vec_1[0] * vec_2[0] \
           + vec_1[1] * vec_2[1] \
           + vec_1[2] * vec_2[2]
    

def spherical_components(vector):
    """
    Return the spherical components of a 3-vector or 4-vector.
    
    Parameters
    ----------
    vector : ndarray of shape (3,) or (4,)
        Vector of which the spherical components are calculated.
        
    Returns
    -------
    v0 : float (optional)
        The time-component if `vector` is a 4-vector.
    v, theta, phi : float
        Spatial spherical components of `vector`.  Here, `v` is the Euclidean
        norm of the spatial part, `theta` is the polar angle and `phi` the
        azimuthal angle.

    Notes
    -----
    This function returns all of the mentioned 'Returns' as a tuple.
    
    """
    
    if len(vector) == 4:
        v0 = vector[0]
        vx = vector[1]
        vy = vector[2]
        vz = vector[3]
    elif len(vector) == 3:
        vx = vector[0]
        vy = vector[1]
        vz = vector[2]
    else:
        raise Exception("`vector` should be an ndarray of shape (3,) or (4,).")
    
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    
    if v != 0:
        theta = np.arccos(vz / v)
        if vx != 0 and vx**2 + vy**2 != 0:
            phi = np.sign(vy) * np.arccos(vx / np.sqrt(vx**2 + vy**2))
        else:
            phi = 0   
    else:
        theta = 0
        phi = 0

    if len(vector) == 4:

        if phi < 0:
            return v0, v, theta, phi + 2 * np.pi
        else:
            return v0, v, theta, phi
        
    elif len(vector) == 3:
    
        if phi < 0:
            return v, theta, phi + 2 * np.pi
        else:
            return v, theta, phi


def cylindrical_components(vector):
    """
    Return the cylindrical components of a 3-vector or 4-vector.
    
    Parameters
    ----------
    vector : ndarray of shape (3,) or (4,)
        Vector of which the cylindrical components are calculated.
        
    Returns
    -------
    v0 : float (optional)
        The time-component if `vector` is a 4-vector.
    rho, phi, vz : float
        `rho` is the Euclidean norm of the projection of the 3-vector onto the
        x,y-plane, `phi` is the standard azimuthal angle and `vz` is the 
        Cartesian z-component of the vector.

    Notes
    -----
    This function returns all of the mentioned 'Returns' as a tuple.
    
    """
    
    if len(vector) == 4:
        v0 = vector[0]
        vx = vector[1]
        vy = vector[2]
        vz = vector[3]
    elif len(vector) == 3:
        vx = vector[0]
        vy = vector[1]
        vz = vector[2]
    else:
        raise Exception("`vector` should be an ndarray of shape (3,) or (4,).")
    
    rho = np.sqrt(vx**2 + vy**2)
    phi = np.arctan2(np.real(vy), np.real(vx))

    if len(vector) == 3:
        return rho, phi, vz
    elif len(vector) == 4:
        return v0, rho, phi, vz


def cartesian_components(vector):
    """
    Return the Cartesian components of a 3-vector or 4-vector.
    
    Parameters
    ----------
    vector : ndarray of shape (3,) or (4,)
        Vector of which the Cartesian components are calculated.
        
    Returns
    -------
    v0 : float (optional)
        The time-component if `vector` is a 4-vector.
    vx, vy, vz : float
        The spatial Cartesian components of `vector`.

    Notes
    -----
    This function returns all of the mentioned 'Returns' as a tuple.
    
    """
    
    if len(vector) == 4:
        v0 = vector[0]
        vx = vector[1]
        vy = vector[2]
        vz = vector[3]
    elif len(vector) == 3:
        vx = vector[0]
        vy = vector[1]
        vz = vector[2]
    else:
        raise Exception("`vector` should be an ndarray of shape (3,) or (4,).")
    
    if len(vector) == 3:
        return vx, vy, vz
    elif len(vector) == 4:
        return v0, vx, vy, vz

