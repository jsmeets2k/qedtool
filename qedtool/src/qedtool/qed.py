import numpy as np
import itertools
import pickle
import time

from .relativity import lorentzian_product, spherical_components, \
                        ThreeVector, FourVector


# 4x4 identity
I4 = np.eye(4)


# Gamma matrices in the chiral basis
GAMMA = np.array([[[0, 0, 1, 0], 
                   [0, 0, 0, 1], 
                   [1, 0, 0, 0], 
                   [0, 1, 0, 0]], 
                  [[0, 0, 0, 1], 
                   [0, 0, 1, 0], 
                   [0, -1, 0, 0], 
                   [-1, 0, 0, 0]],   
                  [[0, 0, 0, -1j], 
                   [0, 0, 1j, 0], 
                   [0, 1j, 0, 0], 
                   [-1j, 0, 0, 0]], 
                  [[0, 0, 1, 0], 
                   [0, 0, 0, -1], 
                   [-1, 0, 0, 0], 
                   [0, 1, 0, 0]]])


# Fine structure and elementary charge in natural units
ALPHA = 1 / 137.035999177
CHARGE = np.sqrt(4 * np.pi * ALPHA)


# Electron, muon and tau masses in MeV
ELECTRON_MASS = 0.511
MUON_MASS = 105.6583755
TAU_MASS = 1776.86


# Dirac spinor boost generators
S01 = 1j * (GAMMA[0].dot(GAMMA[1]) - GAMMA[1].dot(GAMMA[0])) / 4
S02 = 1j * (GAMMA[0].dot(GAMMA[2]) - GAMMA[2].dot(GAMMA[0])) / 4
S03 = 1j * (GAMMA[0].dot(GAMMA[3]) - GAMMA[3].dot(GAMMA[0])) / 4

# Dirac spinor rotation generators
S12 = 1j * (GAMMA[1].dot(GAMMA[2]) - GAMMA[2].dot(GAMMA[1])) / 4
S13 = 1j * (GAMMA[1].dot(GAMMA[3]) - GAMMA[3].dot(GAMMA[1])) / 4
S23 = 1j * (GAMMA[2].dot(GAMMA[3]) - GAMMA[3].dot(GAMMA[2])) / 4



# Dirac spinor and particle classes


class DiracSpinor:
    """
    Class for bispinors, i.e. arrays of shape (4,) with one spin index.

    Parameters
    ----------
    handedness : int
        The handedness of the fermion.  Must be +1 or -1.
    pmu : FourVector
        The 4-momentum of which defines the Dirac spinor.
    anti : Boolean
        Whether the momentum space spinor corresponds to a fermion or
        antifermion.
    adjoint : Boolean
        Whether the Dirac adjoint must be taken.

    Attributes
    ----------
    bispinor : DiracSpinor
        The Dirac spinor.
    adjoint : Boolean
        Whether the Dirac spinor is adjointed or not.

    Methods
    -------
    dirac_adjoint
        Takes the Dirac adjoint of a Dirac spinor.
    
    """

    def __init__(self, handedness, pmu, anti=False, adjoint=False):

        # Check if `pmu` is a 4-vector
        if not isinstance(pmu, FourVector):
            raise TypeError("`pmu` must be of type `FourVector`.")
        
        # Used so that an adjoint cannot be added to a regular bispinor
        self.adjoint = adjoint

        if adjoint == False:
            if anti == False:
                self.bispinor = fermion_polarization(handedness, pmu)
            elif anti == True:
                self.bispinor = antifermion_polarization(handedness, pmu)
            else:
                raise Exception("`anti` must be `True` or `False`.")
        elif adjoint == True:
            if anti == False:
                self.bispinor = dirac_adjoint(
                    fermion_polarization(handedness, pmu)
                    )
            elif anti == True:
                self.bispinor = dirac_adjoint(
                    antifermion_polarization(handedness, pmu)
                    )
            else:
                raise Exception("`anti` must be `True` or `False`.")
        else:
            raise Exception("`adjoint` must be `True` or `False`.")
        
    def __add__(self, other): 

        # Check if `other` is a Dirac spinor
        if not isinstance(other, DiracSpinor):
            return NotImplemented
        
        # Cannot add a Dirac spinor to an adjointed Dirac spinor
        if self.adjoint == True and other.adjoint == False \
        or self.adjoint == False and other.adjoint == True:
            raise Exception("Dirac spinors and adjointed Dirac spinors" \
                            + " cannot be added to each other.")

        # The sum of `self` and `other`
        sspin_bispinor = self.bispinor + other.bispinor

        # Create an instance with an incorrect spinor
        pmu = FourVector(1, 1, 0, 0)
        if self.adjoint == True:
            sspin = DiracSpinor(1, pmu, adjoint=True)
        elif self.adjoint == False:
            sspin = DiracSpinor(1, pmu, adjoint=False)

        # The Dirac spinor as array
        sspin.bispinor = sspin_bispinor

        return sspin
    
    def __sub__(self, other): 

        # Check if `other` is a Dirac spinor
        if not isinstance(other, DiracSpinor):
            return NotImplemented
        
        # Cannot add a Dirac spinor to an adjointed Dirac spinor
        if self.adjoint == True and other.adjoint == False \
        or self.adjoint == False and other.adjoint == True:
            raise Exception("Dirac spinors and adjointed Dirac spinors" \
                            + " cannot be subtracted from each other.")

        # The sum of self and other
        dspin_bispinor = self.bispinor - other.bispinor

        # Create an instance with an incorrect spinor
        pmu = FourVector(1, 1, 0, 0)
        if self.adjoint == True:
            dspin = DiracSpinor(1, pmu, adjoint=True)
        elif self.adjoint == False:
            dspin = DiracSpinor(1, pmu, adjoint=False)

        # The Dirac spinor as array
        dspin.bispinor = dspin_bispinor

        return dspin
    
    def __mul__(self, other): 

        # Check if `other` is a scalar
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, complex) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64) \
                or isinstance(other, np.complex128)):
            return NotImplemented

        # The product of self and other
        pspin_bispinor = self.bispinor * other

        # Create an instance with an incorrect spinor
        pmu = FourVector(1, 1, 0, 0)
        if self.adjoint == True:
            pspin = DiracSpinor(1, pmu, adjoint=True)
        elif self.adjoint == False:
            pspin = DiracSpinor(1, pmu, adjoint=False)

        # The Dirac spinor as array
        pspin.bispinor = pspin_bispinor

        return pspin
    
    def __rmul__(self, other): 

        # Check if `other` is a scalar
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, complex) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64) \
                or isinstance(other, np.complex128)):
            return NotImplemented

        # The product of self and other
        pspin_bispinor = self.bispinor * other

        # Create an instance with an incorrect spinor
        pmu = FourVector(1, 1, 0, 0)
        if self.adjoint == True:
            pspin = DiracSpinor(1, pmu, adjoint=True)
        elif self.adjoint == False:
            pspin = DiracSpinor(1, pmu, adjoint=False)

        # The Dirac spinor as array
        pspin.bispinor = pspin_bispinor

        return pspin
    
    def __truediv__(self, other): 

        # Check if `other` is a scalar
        if not (isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, complex) \
                or isinstance(other, np.int32) \
                or isinstance(other, np.float64) \
                or isinstance(other, np.complex128)):
            return NotImplemented

        # The product of self and other
        pspin_bispinor = self.bispinor / other

        # Create an instance with an incorrect spinor
        pmu = FourVector(1, 1, 0, 0)
        if self.adjoint == True:
            pspin = DiracSpinor(1, pmu, adjoint=True)
        elif self.adjoint == False:
            pspin = DiracSpinor(1, pmu, adjoint=False)

        # The Dirac spinor as array
        pspin.bispinor = pspin_bispinor

        return pspin
    
    def __neg__(self): 

        # Check if `self` is a Dirac spinor
        if not isinstance(self, DiracSpinor):
            return NotImplemented

        # Minus `self` bispinor
        nspin_bispinor = -self.bispinor

        # Create an instance with an incorrect spinor
        pmu = FourVector(1, 1, 0, 0)
        if self.adjoint == True:
            nspin = DiracSpinor(1, pmu, adjoint=True)
        elif self.adjoint == False:
            nspin = DiracSpinor(1, pmu, adjoint=False)

        # The Dirac spinor as array
        nspin.bispinor = nspin_bispinor

        return nspin
    
    def dirac_adjoint(psi):

        # Check if `psi` is of type `DiracSpinor`
        if not isinstance(psi, DiracSpinor):
            raise TypeError("`psi` must be of type `DiracSpinor`.")

        # Overwrite the `bispinor` and `adjoint` attributes
        psi.bispinor = dirac_adjoint(psi.bispinor)
        if psi.adjoint == False:
            psi.adjoint = True
        elif psi.adjoint == True:
            psi.adjoint = False
        
        return psi


class RealParticle:
    """
    Class for real particles, i.e. measurable particles that enter or exit the 
    scattering event.

    Parameters
    ----------
    species : str
        The type of particle.  Must be `electron`, `positron` or `photon`.
    handedness : int
        The handedness of the fermion.  Must be +1 or -1.
    pmu : FourVector
        The on-shell 4-momentum of the particle.
    direction : str
        Whether the particle enters or exits the scattering event.  Must be
        'in' or 'out'.

    Attributes
    ----------
    species : str
        The type of particle.
    pmu : FourVector
        The on-shell 4-momentum of the particle.
    direction : str
        Whether the particle enters or exits the scattering event.
    spin : float
        The intrinsic spin of the particle.
    mass : float
        The mass of the particle in MeV.
    charge : float
        The electric charge of the particle expressed in elementary charges.
    polarization : DiracSpinor or FourVector
        The polarization of the particle.  For fermions, `polarization` is of
        type `DiracSpinor`.  For photons, it is of type `FourVector`.

    Methods
    -------
    electron
        Real electron with a definite momentum and handedness.
    positron
        Real positron with a definite momentum and handedness.
    photon
        Real photon with a definite momentum and handedness.
    muon
        Real muon with a definite momentum and handedness.
    antimuon
        Real antimuon with a definite momentum and handedness.
    
    """

    def __init__(self, species, handedness, pmu, direction):

        # Specifies the type of particle
        self.species = species

        # Particle's 4-momentum (FourVector class)
        if isinstance(pmu, FourVector):
            self.four_momentum = pmu
        else:
            raise Exception("`pmu` must be of type `FourVector`.")

        # Whether the particle enters or exits the scattering
        if direction == 'in' or direction == 'out':
            self.direction = direction
        else:
            raise Exception("`direction` must be `'in'` or `'out'`.")

        # Yet to be determined parameters
        self.spin = None
        self.mass = None
        self.charge = None
        self.polarization = None

    def electron(handedness, pmu, direction):
        """
        Return an external electron.

        Parameters
        ----------
        handedness : int
            The handedness of the electron.  Must be +1 or -1.
        pmu : FourVector
            The on-shell 4-momentum of the electron.  Must be time-like.
        direction : str
            Whether the electron enters or exits the scattering event.  Must 
            be 'in' or 'out'.

        Returns
        -------
        pcl : RealParticle
            The real electron.

        Notes
        -----
        If the electron exits the scattering event, then the Dirac adjoint of
        its Dirac spinor is automatically taken.
        
        """

        # Check handedness
        if handedness == 1 or handedness == -1:
            pass
        else:
            raise Exception("`handedness` must be +1 or -1.")
        
        # Make an instance
        pcl = RealParticle('electron', handedness, pmu, direction)

        # Check whether `pmu` is time-like
        t = pmu.sphericals[0]
        v = pmu.sphericals[1]
        if not t >= v:
            raise Exception(
                "`pmu` must be time-like for massive particles " \
                + "(or light-like in the ultrarelativistic regime)."
                )

        # Charge, spin and bare mass
        pcl.charge = -1.0
        pcl.spin = 1/2
        pcl.mass = ELECTRON_MASS

        # Electron's polarization
        if direction == 'in':
            pcl.polarization = DiracSpinor(handedness, pmu, anti=False,
                                           adjoint=False)
        elif direction == 'out':
            pcl.polarization = DiracSpinor(handedness, pmu, anti=False,
                                           adjoint=True)

        return pcl
    
    def positron(handedness, pmu, direction):
        """
        Return an external positron.

        Parameters
        ----------
        handedness : int
            The handedness of the positron.  Must be +1 or -1.
        pmu : FourVector
            The on-shell 4-momentum of the positron.  Must be time-like.
        direction : str
            Whether the positron enters or exits the scattering event.  Must 
            be 'in' or 'out'.

        Returns
        -------
        pcl : RealParticle
            The real positron.

        Notes
        -----
        If the positron enters the scattering event, then the Dirac adjoint of
        its Dirac spinor is automatically taken.
        
        """

        # Check handedness
        if handedness == 1 or handedness == -1:
            pass
        else:
            raise Exception("`handedness` must be +1 or -1.")
        
        # Make an instance
        pcl = RealParticle('positron', handedness, pmu, direction)

        # Check whether `pmu` is time-like
        t = pmu.sphericals[0]
        v = pmu.sphericals[1]
        if t >= v:
            pcl.four_momentum = pmu
        else:
            raise Exception(
                "`pmu` must be time-like for massive particles " \
                + "(or light-like in the ultrarelativistic regime)."
                )

        # Charge, spin and bare mass
        pcl.charge = 1.0
        pcl.spin = 1/2
        pcl.mass = ELECTRON_MASS

        # Particle's polarization
        if direction == 'in':
            pcl.polarization = DiracSpinor(handedness, pmu, anti=True,
                                           adjoint=True)
        elif direction == 'out':
            pcl.polarization = DiracSpinor(handedness, pmu, anti=True,
                                           adjoint=False)

        return pcl
    
    def photon(handedness, pmu, direction):
        """
        Returns an external photon.

        Parameters
        ----------
        handedness : int
            The handedness of the photon.  Must be +1 or -1.
        pmu : FourVector
            The on-shell 4-momentum of the photon.  Must be light-like.
        direction : str
            Whether the photon enters or exits the scattering event.  Must 
            be 'in' or 'out'.

        Returns
        -------
        pcl : RealParticle
            The real photon.

        Notes
        -----
        If the photon exits the scattering event, then the complex conjugate of
        its 4-polarization is automatically taken.
        
        """

        # Check handedness
        if handedness == 1 or handedness == -1:
            pass
        else:
            raise Exception("`handedness` must be +1 or -1.")
        
        # Make an instance
        pcl = RealParticle('photon', handedness, pmu, direction)

        # Check whether `pmu` is light-like
        if np.round(lorentzian_product(pmu.vector, pmu.vector), 7) == 0.0:
            pcl.four_momentum = pmu
        else:
            raise Exception("`pmu` must be light-like for photons. " \
                            + "A virtuality of at least 1e-7 is required.")

        # Charge, spin and bare mass
        pcl.charge = 0.0
        pcl.spin = 1.0
        pcl.mass = 0.0

        # Photon's polarization
        if direction == 'in':
            pcl.polarization = FourVector.polarization(handedness, pmu)
        elif direction == 'out':
            pcl.polarization = FourVector.polarization(handedness, pmu, 
                                                       conjugate=True)

        return pcl
    
    def muon(handedness, pmu, direction):
        """
        Return an external muon.

        Parameters
        ----------
        handedness : int
            The handedness of the muon.  Must be +1 or -1.
        pmu : FourVector
            The on-shell 4-momentum of the muon.  Must be time-like.
        direction : str
            Whether the muon enters or exits the scattering event.  Must 
            be 'in' or 'out'.

        Returns
        -------
        pcl : RealParticle
            The real muon.

        Notes
        -----
        If the muon exits the scattering event, then the Dirac adjoint of
        its Dirac spinor is automatically taken.
        
        """

        # Check handedness
        if handedness == 1 or handedness == -1:
            pass
        else:
            raise Exception("`handedness` must be +1 or -1.")
        
        # Make an instance
        pcl = RealParticle('muon', handedness, pmu, direction)

        # Check whether `pmu` is time-like
        t = pmu.sphericals[0]
        v = pmu.sphericals[1]
        if t >= v:
            pcl.four_momentum = pmu
        else:
            raise Exception(
                "`pmu` must be time-like for massive particles " \
                + "(or light-like in the ultrarelativistic regime)."
                )

        # Charge, spin and bare mass
        pcl.charge = -1.0
        pcl.spin = 1/2
        pcl.mass = MUON_MASS

        # Muon's polarization
        if direction == 'in':
            pcl.polarization = DiracSpinor(handedness, pmu, anti=False,
                                           adjoint=False)
        elif direction == 'out':
            pcl.polarization = DiracSpinor(handedness, pmu, anti=False,
                                           adjoint=True)

        return pcl
  
    def antimuon(handedness, pmu, direction):
        """
        Return an external antimuon.

        Parameters
        ----------
        handedness : int
            The handedness of the antimuon.  Must be +1 or -1.
        pmu : FourVector
            The on-shell 4-momentum of the antimuon.  Must be time-like.
        direction : str
            Whether the antimuon enters or exits the scattering event.  Must 
            be 'in' or 'out'.

        Returns
        -------
        pcl : RealParticle
            The real antimuon.

        Notes
        -----
        If the positron enters the scattering event, then the Dirac adjoint of
        its Dirac spinor is automatically taken.
        
        """

        # Check handedness
        if handedness == 1 or handedness == -1:
            pass
        else:
            raise Exception("`handedness` must be +1 or -1.")
        
        # Make an instance
        pcl = RealParticle('antimuon', handedness, pmu, direction)

        # Check whether `pmu` is time-like
        t = pmu.sphericals[0]
        v = pmu.sphericals[1]
        if t >= v:
            pcl.four_momentum = pmu
        else:
            raise Exception(
                "`pmu` must be time-like for massive particles " \
                + "(or light-like in the ultrarelativistic regime)."
                )

        # Charge, spin and bare mass
        pcl.charge = 1.0
        pcl.spin = 1/2
        pcl.mass = MUON_MASS

        # Antimuon's polarization
        if direction == 'in':
            pcl.polarization = DiracSpinor(handedness, pmu, anti=True,
                                           adjoint=True)
        elif direction == 'out':
            pcl.polarization = DiracSpinor(handedness, pmu, anti=True,
                                           adjoint=False)

        return pcl
 

class VirtualParticle:
    """
    Class for virtual particles, i.e. immeasurable field configurations.

    Parameters
    ----------
    species : str
        The type of particle.  Must be 'fermion' or 'photon'.
    pmu : FourVector
        The (off-shell) 4-momentum of the particle.

    Attributes
    ----------
    species : str
        The type of particle.
    four_momentum : FourVector
        The on-shell 4-momentum of the particle.
    virtuality : float
        The squared Lorentzian norm of `pmu`.
    propagator : ndarray
        The field's propagator.

    Methods
    -------
    electron
        Virtual electron.
    positron
        Virtual positron.
    photon
        Virtual photon.
    muon
        Virtual muon.
    antimuon
        Virtual antimuon.
    
    """

    def __init__(self, species, pmu):

        # Particle's 4-momentum (FourVector class)
        if isinstance(pmu, FourVector):
            self.four_momentum = pmu
        else:
            raise Exception("`pmu` must be of type `FourVector`.")
        
        # Specifies the type of particle
        self.species = species

        # Yet to be determined parameters
        self.virtuality = -lorentzian_product(pmu.vector, pmu.vector)
        self.propagator = None
        self.mass = None

    def electron(pmu):
        """
        Return a virtual electron.

        Parameters
        ----------
        pmu : FourVector
            The (off-shell) 4-momentum of the electron.

        Returns
        -------
        electron : VirtualParticle
            The virtual electron.
        
        """

        # Make an instance
        electron = VirtualParticle('electron', pmu)

        # Assign bare mass and propagator
        electron.mass = ELECTRON_MASS
        electron.propagator = fermion_propagator(pmu, electron.mass)

        return electron
    
    def positron(pmu):
        """
        Return a virtual positron.

        Parameter
        ---------
        pmu : FourVector
            The (off-shell) 4-momentum of the positron.

        Returns
        -------
        positron : VirtualParticle
            The virtual positron.
        
        """

        # Make an instance
        positron = VirtualParticle('positron', pmu)

        # Assign bare mass and propagator
        positron.mass = ELECTRON_MASS
        positron.propagator = fermion_propagator(pmu, positron.mass)

        return positron
    
    def photon(pmu):
        """
        Return a virtual photon.

        Parameter
        ---------
        pmu : FourVector
            The (off-shell) 4-momentum of the photon.

        Returns
        -------
        photon : VirtualParticle
            The virtual photon.
        
        """

        # Make an instance
        photon = VirtualParticle('photon', pmu)

        # Assign bare mass and propagator
        photon.mass = 0.0
        photon.propagator = photon_propagator(pmu)

        return photon

    def muon(pmu):
        """
        Return a virtual muon.

        Parameters
        ----------
        pmu : FourVector
            The (off-shell) 4-momentum of the muon.

        Returns
        -------
        muon : VirtualParticle
            The virtual muon.
        
        """

        # Make an instance
        muon = VirtualParticle('muon', pmu)

        # Assign bare mass and propagator
        muon.mass = MUON_MASS
        muon.propagator = fermion_propagator(pmu, muon.mass)

        return muon

    def antimuon(pmu):
        """
        Return a virtual antimuon.

        Parameters
        ----------
        pmu : FourVector
            The (off-shell) 4-momentum of the antimuon.

        Returns
        -------
        antimuon : VirtualParticle
            The virtual antimuon.
        
        """

        # Make an instance
        antimuon = VirtualParticle('antimuon', pmu)

        # Assign bare mass and propagator
        antimuon.mass = MUON_MASS
        antimuon.propagator = fermion_propagator(pmu, antimuon.mass)



# Polarizations


def fermion_polarization(handedness, pmu):
    """
    Return an on-shell momentum and handedness eigenbispinor of a fermion in
    momentum space.
    
    Parameters
    ----------
    handedness : int
        The handedness of the fermion.  Must be either -1 or +1.
    pmu : ndarray of shape (4,)
        The on-shell 4-momentum of the fermion.
        
    Returns
    -------
    psi : ndarray of shape (4,)
        Fermion bispinor with the specified `handedness` and 4-momentum `pmu`.
    
    """
    
    if isinstance(pmu, np.ndarray):
        p0, p, theta, phi = spherical_components(pmu)
    elif isinstance(pmu, FourVector):
        p0, p, theta, phi = pmu.sphericals
    
    # Errors for handedness
    if handedness != -1 and handedness != 1:
        raise ValueError("handedness must be either +1 or -1.")
    
    if handedness == 1:
        xi = [np.cos(theta/2), np.exp(1j*phi) * np.sin(theta/2)]

    elif handedness == -1:
        xi = [-np.exp(-1j*phi) * np.sin(theta/2), np.cos(theta/2)]

    return np.array([np.sqrt(p0 - handedness * p) * xi[0],
                     np.sqrt(p0 - handedness * p) * xi[1],
                     np.sqrt(p0 + handedness * p) * xi[0],
                     np.sqrt(p0 + handedness * p) * xi[1]])


def antifermion_polarization(handedness, pmu):
    """
    Return an on-shell momentum and handedness eigenbispinor of an antifermion 
    in momentum space.
    
    Parameters:
    -----------
    handedness : int
        The handedness of the antifermion.  Must be either -1 or +1.
    pmu : ndarray of shape (4,)
        The 4-momentum of the antifermion.
        
    Returns:
    --------
    u : ndarray of shape (4,)
        Antifermion bispinor with the specified `handedness` and 4-momentum 
        `pmu`.  If the Euclidean norm of the spatial components of `pmu` is 
        zero, then spin eigenstates with the z-axis as the quantization axis 
        are returned.
    
    """
    
    if isinstance(pmu, np.ndarray):
        p0, p, theta, phi = spherical_components(pmu)
    elif isinstance(pmu, FourVector):
        p0, p, theta, phi = pmu.sphericals
    
    # Errors for handedness
    if handedness != -1 and handedness != 1:
        raise ValueError("handedness must be either +1 or -1.")
    
    if handedness == -1:
        xi = [np.cos(theta/2), np.exp(1j*phi) * np.sin(theta/2)]

    elif handedness == 1:
        xi = [-np.exp(1j*phi) * np.sin(theta/2), np.cos(theta/2)]

    return np.array([
        - handedness * np.sqrt(p0 + handedness * p) * xi[0],
        - handedness * np.sqrt(p0 + handedness * p) * xi[1],
        handedness * np.sqrt(p0 - handedness * p) * xi[0],
        handedness * np.sqrt(p0 - handedness * p) * xi[1]
    ])
    

def photon_polarization(handedness, pmu):
    """ 
    Return an on-shell handedness photon eigenpolarization.
    
    Parameters
    ----------
    handedness : int
        handedness of the photon.  Must be -1 or +1.
    pmu : ndarray of shape (4,)
        4-momentum of the photon.  Must be light-like.
    
    Returns
    -------
    eps : ndarray of shape (4,)
        The 4-polarization of the photon, a momentum and handedness 
        eigenpolarization.
    
    """

    # Also include the possibility for `pmu` to be a `FourVector`
    if isinstance(pmu, np.ndarray):
        pass
    elif isinstance(pmu, FourVector):
        pmu = pmu.vector
    
    # Retrieve spherical components
    theta, phi = spherical_components(pmu)[2:4]
    
    # Errors for handedness
    if handedness != -1 and handedness != 1:
        raise ValueError("handedness must be either +1 or -1.")
    
    if handedness == 1 or handedness == -1:
        return np.array([0,
                         np.cos(theta) * np.cos(phi) \
                            - np.sign(handedness) * 1j * np.sin(phi),
                         np.cos(theta) * np.sin(phi) \
                            + np.sign(handedness) * 1j * np.cos(phi),
                         -np.sin(theta)]) / np.sqrt(2)
        


# Propagators


def fermion_propagator(pmu, m):
    """
    Return the bare momentum space propagator of an off-shell (anti)fermion.
    
    Parameters
    ----------
    pmu : ndarray of shape (4,)
        4-momentum of the (anti)fermion.  Must be off-shell.
    m : float
        Bare mass of the (anti)fermion, not to be confused with the 
        square root of its virtuality.
        
    Returns
    -------
    g0 : ndarray of shape (4, 4)
        Bare momentum space propagator of the (anti)fermion.  Both indices
        are spin indices.
    
    """

    # Also include the possibility for `pmu` to be a `FourVector`
    if isinstance(pmu, np.ndarray):
        pass
    elif isinstance(pmu, FourVector):
        pmu = pmu.vector
    
    # Error for pmu
    if len(pmu) != 4:
        raise Exception("pmu must have shape (4,).")
        
    # Errors for m
    if m < 0:
        raise Exception("m must be positive.")
    
    mass = np.array([[m, 0, 0, 0],
                     [0, m, 0, 0],
                     [0, 0, m, 0], 
                     [0, 0, 0, m]])
    
    return 1j * (slashed(pmu) + mass) / (lorentzian_product(pmu, pmu) - m**2)


def photon_propagator(pmu):
    """
    Return the bare momentum space photon propagator in the Feynman gauge.
    
    Parameters
    ----------
    pmu : ndarray of shape (4,)
        Photon's 4-momentum.  Must be off-shell.
        
    Returns
    -------
    d0 : ndarray of shape (4, 4)
        Bare momentum space photon propagator.  Both indices are spin 
        indices.
    
    """

    # Also include the possibility for `pmu` to be a `FourVector`
    if isinstance(pmu, np.ndarray):
        pass
    elif isinstance(pmu, FourVector):
        pmu = pmu.vector
    
    # Error for pmu
    if len(pmu) != 4:
        raise Exception("pmu must have shape (4,).")
    
    return (-1j / (lorentzian_product(pmu, pmu))) \
            * np.array([[1, 0, 0, 0], 
                        [0, -1, 0, 0], 
                        [0, 0, -1, 0], 
                        [0, 0, 0, -1]])



# Miscellaneous functions


def constant(quantity, units=None):
    """
    Returns constants in the desired units.

    Parameters
    ----------
    quantity : str
        The quantity.
    units : str
        The units in which to express `quantity`, which range from `'eV'` to
        `'GeV'`.

    Returns
    -------
    quantity_in_units : float
        The value of `quantities` expressed in the specified `units`.
    
    """

    if quantity == 'electron mass' or quantity == 'positron mass':
        if units == 'eV':
            globals()['ELECTRON_MASS'] = 511000
            return 511000
        elif units == 'keV':
            globals()['ELECTRON_MASS'] = 511000 * 1e-3
            return 511000 * 1e-3
        elif units == 'MeV' or units == None:
            globals()['ELECTRON_MASS'] = 511000 * 1e-6
            return 511000 * 1e-6
        elif units == 'GeV':
            globals()['ELECTRON_MASS'] = 511000 * 1e-9
            return 511000 * 1e-9
        else:
            raise ValueError("Invalid units.")
               
    elif quantity == 'muon mass' or quantity == 'antimuon mass':
        if units == 'eV':
            globals()['MUON_MASS'] = 105658375.5
            return 105658375.5
        elif units == 'keV':
            globals()['MUON_MASS'] = 105658375.5 * 1e-3
            return 105658375.5 * 1e-3
        elif units == 'MeV' or units == None:
            globals()['MUON_MASS'] = 105658375.5 * 1e-6
            return 105658375.5 * 1e-6
        elif units == 'GeV':
            globals()['MUON_MASS'] = 105658375.5 * 1e-9
            return 105658375.5 * 1e-9
        else:
            raise ValueError("Invalid units.")
        
    elif quantity == 'charge' or quantity == 'elementary charge':
        if units == None:
            return CHARGE
        else:
            raise Exception(
                "Units cannot be specified for the elementary charge."
                )
        
    elif quantity == 'fine structure' or \
        quantity == 'fine structure constant' or quantity == 'alpha':
        if units == None:
            return ALPHA
        else:
            raise Exception(
                "Units cannot be specified for the fine structure constant."
                )
        
    elif quantity == 'vacuum speed of light' or quantity == 'c':
        if units == None:
            return 1
        else:
            raise Exception(
                "Units cannot be specified for the vacuum speed of light."
                )


def slashed(vmu):
    """
    Return the contraction of a 4-vector with the gamma matrices.
    
    Parameters
    ----------
    vmu : ndarray or FourVector
        4-vector that will be contracted with the gamma matrices.
        
    Returns
    -------
    vs : ndarray of shape (4, 4)
        Contraction of `vmu` with the gamma matrices.
    
    """
    
    if isinstance(vmu, np.ndarray):
        
        if len(vmu) != 4:
            raise Exception("`vmu` must have shape (4,).")
    
        return vmu[0] * GAMMA[0] \
               - vmu[1] * GAMMA[1] \
               - vmu[2] * GAMMA[2] \
               - vmu[3] * GAMMA[3]
    
    elif isinstance(vmu, FourVector):

        return vmu.vector[0] * GAMMA[0] \
               - vmu.vector[1] * GAMMA[1] \
               - vmu.vector[2] * GAMMA[2] \
               - vmu.vector[3] * GAMMA[3]
    
    else:

        raise TypeError("`vmu` must be of type `ndarray` of `FourVector`.")


def dirac_adjoint(psi):
    """
    Return the Dirac adjoint of a bispinor.
    
    Parameters
    ----------
    psi : DiracSpinor or ndarray of shape (4,)
        Dirac spinor of which the Dirac adjoint will be taken.
    
    Returns
    -------
    psi_bar : ndarray of shape (4,)
        The Dirac adjoint of `psi`.
    
    """

    # Also include the possibility for `psi` to be a `DiracSpinor`
    if isinstance(psi, np.ndarray):
        pass
    elif isinstance(psi, DiracSpinor):
        psi = psi.bispinor
    
    # Error for psi
    if len(psi) != 4:
        raise Exception("`psi` must have shape (4,).")
    
    return np.dot(np.conjugate(psi), np.array([[0, 0, 1, 0], 
                                               [0, 0, 0, 1], 
                                               [1, 0, 0, 0], 
                                               [0, 1, 0, 0]]))


def dirac_current(psi_1, psi_2):
    """
    Return the momentum space Dirac current from on-shell bispinors.
    
    Parameters
    ----------
    psi_1 : ndarray of shape (4,)
        The first Dirac adjointed on-shell Dirac spinor.
    psi_2 : ndarray of shape (4,)
        Second on-shell Dirac spinor.
        
    Returns
    -------
    jmu : ndarray of shape (4,)
        The momentum space Dirac current, a 4-vector.
    
    """

    # Also include the possibility for `psi_1` to be a `DiracSpinor`
    if isinstance(psi_1, np.ndarray):
        pass
    elif isinstance(psi_1, DiracSpinor):
        psi_1 = psi_1.bispinor

    # Also include the possibility for `psi_2` to be a `DiracSpinor`
    if isinstance(psi_2, np.ndarray):
        pass
    elif isinstance(psi_2, DiracSpinor):
        psi_2 = psi_2.bispinor
    
    return np.array([np.dot(psi_1, np.dot(GAMMA[0], psi_2)),
                     np.dot(psi_1, np.dot(GAMMA[1], psi_2)),
                     np.dot(psi_1, np.dot(GAMMA[2], psi_2)),
                     np.dot(psi_1, np.dot(GAMMA[3], psi_2))])


def handedness_config(n, fixed=None, fixedval=None):
    """
    Return an array that contains all possible handedness combinations.  User
    may specify fixed helicities.

    Parameters
    ----------
    n : int
        Number of helicities.  Must be positive.
    fixed : array_like or int, optional
        1D array that contains the indices of fixed helicities.  By default,
        no helicities are fixed.  The length of `fixed` must be less than `n`.
    fixedval : array_like or int, optional
        1D array that contains the fixed helicities.  It is a 1-to-1 mapping
        to `fixed`.  If `fixed` is specified, then `fixedval` must also be 
        specified.  The length of `fixedval` must be equal to the length of 
        `fixed`.

    Returns
    -------
    helicities : ndarray
        2D array that contains all combinations of helicities.

    """
    
    # Errors for n
    if type(n) != int:
        raise TypeError("`n` must be of type int.")
    if n <= 0:
        raise Exception("`n` must be larger than zero.")
        
    # If fixed and fixedval are not None
    if fixed != None:
        
        # Check if fixedval is defined
        if fixedval == None:
            raise Exception("If `fixed` is specified " \
                            + "then `fixedval` also needs to be specified.")
        
        # Make sure fixed and fixedval are of equal length
        if len(fixed) != len(fixedval):
            raise Exception("`fixed` and `fixedval` should " \
                            + "be of equal length.")

        # Errors for fixed
        for i in range(len(fixed)):
            if type(fixed[i]) != int:
                raise TypeError("Elements of `fixed` should be of type `int`.")
            if fixed[i] < 0 or fixed[i] > n - 1:
                raise Exception("Elements of `fixed` should be in the " \
                                + "interval [0, n-1].")
                
        # Errors for fixedval
        for i in range(len(fixedval)):
            if type(fixedval[i]) != int:
                raise TypeError("Elements of `fixedval` should " \
                                + "be of type int.")
            if fixedval[i] != 1 and fixedval[i] != -1:
                raise Exception("Elements of fixedval should be " \
                                + "either +1, or -1.")
        
    # Generate all possibilities and empty handedness configuration list
    lst = list(itertools.product([1, -1], repeat=n))
    h = np.zeros((2**n, n))
    
    if fixed == None:
        
        return lst
    
    elif type(fixed) == int and type(fixedval) == int:
        
        # Filling the handedness configuration list
        for i in range(len(h)):
            for j in range(len(h[i])):
                h[i][j] = lst[i][j]
        
        # Fixing the selected helicities
        for i in range(len(h)):
            h[i][fixed] = fixedval
        
        # Removing duplicates
        helicities = np.unique(h, axis=0)
        
        return np.array(helicities)
    
    else:
        
        # Filling the handedness configuration list
        for i in range(len(h)):
            for j in range(len(h[i])):
                h[i][j] = lst[i][j]
        
        # Filling in the fixed values
        for i in range(len(h)):
            for j in range(len(fixedval)):
                h[i][fixed[j]] = fixedval[j]
        
        # Removing duplicates
        helicities = np.unique(h, axis = 0)
        
        return helicities.astype(int)


def empty_lists(n):
    """
    Return a list with empty list.

    Parameters
    ----------
    n : int
        The number of empty lists inside of the main list.

    Returns
    -------
    lists : list
        2D array, an array filled with `n` empty lists.
    
    """
    
    lists = []
    for i in range(n):
        lists.append([])
        
    return lists


def progress(idx, length):
    """
    Show the percentage of the for-loop that is completed.

    Parameters
    ----------
    idx : int
        The current value of the index that is being looped over.
    length : int
        The length of the array over which the for-loop runs.
    
    """
    
    progress = str(np.round(100 * (idx + 1) / length, 3)) + '%'
    print(progress, end="\r")


def save_data(filename, keys, data):
    """
    Save a dictionary with specified keys and values under the specified 
    filename.

    Parameters
    ----------
    filename : str
        The name of the file to be saved.
    keys : array_like
        1D array containing the keys for the dictionary.
    data : array_like
        Array containing the data that correspond to `keys`.
    
    """
    
    # Check is `keys` is a 1D array
    for key in keys:
        if isinstance(key, np.ndarray) or isinstance(key, list):
            raise Exception("`keys` must be a 1D array.")

    # Initialize dictionary
    dictionary = dict.fromkeys(keys)
    
    # Fill the dictionary
    for i in range(len(keys)):
        dictionary[keys[i]] = np.array(data[i])
        
    # Open file
    file = open(filename + '.pkl', 'wb')

    # Write dictionary to file
    pickle.dump(dictionary, file)

    # Close file
    file.close()

