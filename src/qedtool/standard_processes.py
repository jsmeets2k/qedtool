import numpy as np
from .qed import *
from .qinfo import *


# Import electron charge value:
e = constant("elementary charge")

# Import electron and muon mass
m_e = constant("electron mass")
m_mu = constant("muon mass")


def standard_scattering(in_state, scattering, momentum, 
            theta, phi=None, filename=None, projection=None, dp=False, 
            dcs=False, c=False, stokes=False, deg_pol=False, amplitudes=False, 
            out_state=False):
    
    """
    Calculates differential probability, differential cross section, 
    output state concurrence, Stokes parameters, degree of polarization, 
    scattering amplitudes, and the outgoing helicity-basis states for one of 
    six standard QED scattering processes in the centre-of-mass frame at tree 
    level. The available processes are: Compton, Bhabha, Moller, electron-muon 
    scattering, electron-positron annihilation, and electron-positron to 
    muon-antimuon.

    Parameters
    -------
    in_state : QuantumState
        QuantumState instance for the quantum state of incoming 
        electron, photon, or muon particles for different incoming centre-of-
        mass momentum magnitude values. Of the same length as the `momentum`
        input.
    scattering : str
        Determines which scattering processes calculations are computed for. 
        One of the following processes: 'compton', 'bhabha', 'moller', 
        'electron_positron_annihilation', 
        'electron_muon', 'electron_positron_to_muon_antimuon'.
    momentum : array_like 
        Values of the incoming particles' three-momentum magnitude (in the 
        centre-of-mass frame) over which all output quantities are calculated.
    theta : array_like 
        Values of the polar angle (angle between the outgoing and incoming 
        electron three-momenta) over which all output quantities are 
        calculated.
    phi : array_like 
        Values of the azimuthal angle (angle between the x-axis of the 
        scattering coordinate system and the projection of the outgoing 
        electron's three-momentum onto the x-y plane) over which all output 
        quantities are calculated.
    filename : str 
        If `filename` is `str` the output data dictionary is saved as a 
        Pickle file, with the string as a title. `None` by default.
    projection : QuantumState 
        Optional input of the state onto which the outgoing (scattered) 
        electron-photon state is projected, such that differential proobability
        values are adjusted accordingly. `False` by default.
    dp : bool 
        If `dp` is `True` an array of values of the spin-averaged 
        sum of |M|^2 (the squared magnitude of amplitudes) is output. The array 
        runs over all input `momentum`, `theta`, and `phi` values. `False` by 
        default.
    dcs : bool
        If `dcs` is `True` an array of values of the differential cross section
        associated with the scattering process is output. The array 
        runs over all input `momentum`, `theta`, and `phi` values. `False` by 
        default.
    c : bool 
        If `c` is `True` an array of concurrence values of the 
        scattered state is output. The array runs over all input `momentum`, 
        `theta`, and `phi` values. `False` by default.
    stokes : bool 
        If `stokes` is `True` an array of the Stokes parameters of 
        the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    deg_pol : bool 
        If `deg_pol` is `True` an array of the degree of polarization 
        values of the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    amplitudes : bool
        If `amplitudes` is `True` arrays of scattering amplitude
        values are output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    out_state : bool
        If `out_state` is `True` an array of outgoing states (as QuantumState
        class instances) is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.

    Returns
    -------
    output_dictionary : dict
        Dictionary of `dp`, `c`, `dcs`, `stokes`, `deg_pol`, `amplitudes`, and
        `out_state` arrays.
    ---------------------------------------------------------------------------
    
    """

    # Raise a type error if in_state and/or 
    # projection state inputs are not of `QuantumState` type:    
    if not isinstance(in_state, QuantumState):
        raise TypeError("Expected 'in_state' to be of type"\
            + f" `QuantumState`, but got {type(in_state).__name__}.")
    
    if not np.shape(in_state.rho) == (4, 4):
        raise TypeError("Expected 'projection' to be"\
            + f" of shape (4,4), but got {np.shape(in_state.rho)}.")
    
    if projection is not None:
        if not isinstance(projection, QuantumState):
            raise TypeError("Expected 'projection' to be of type"\
            + f" `QuantumState`, but got {type(projection).__name__}.")
        
        if np.shape(projection.rho) != (4, 4):
            raise TypeError("Expected 'projection' to be"\
            + f" of shape (4,4), but got {np.shape(projection.rho)}.")

    # Raise a type error if `momentum`, `theta`, and `phi` are not of lists
    # or ndarrays:
    if momentum is not None:
        if not isinstance(momentum, (list, np.ndarray)):
            raise TypeError("Expected `momentum` to be a list or an ndarray,"\
                        + f" but got {type(momentum).__name__}.")
        
    elif momentum == None:
        momentum = [0]
        
    if theta is not None:
        if not isinstance(theta, (list, np.ndarray)):
            raise TypeError("Expected `theta` to be a list or an ndarray,"\
                        + f" but got {type(theta).__name__}.")
        
    elif theta == None:
        theta = [0]
    
    if phi is not None:
        if not isinstance(phi, (list, np.ndarray)):
            raise TypeError(f"Expected `phi` to be a list or an ndarray"\
                        + f", but got {type(phi).__name__}.")
    elif phi == None:
        phi = [0]
    
    # Raise error if dp, c, stokes, deg_pol, are not Boolean:
    if not isinstance(dp, bool):
        raise ValueError("dp must be Boolean (True or False)")
    
    if not isinstance(c, bool):
        raise ValueError("c must be Boolean (True or False)")
    
    if not isinstance(stokes, bool):
        raise ValueError("stokes must be Boolean (True or False)")

    if not isinstance(deg_pol, bool):
        raise ValueError("deg_pol must be Boolean (True or False)")
    
    scatter_names = ['compton', 'bhabha', 'moller', \
        'electron_positron_annihilation', \
        'electron_muon', 'electron_positron_to_muon_antimuon']
    
    if scattering not in scatter_names:
        raise ValueError(f'Invalid value: scattering = {scattering}.' \
                         f'Expected one of the following: {scatter_names}')
    
    # Convert input array_like objects to ndarrays:
    momentum = np.array(momentum)
    theta = np.array(theta)
    phi = np.array(phi)

    # Raise a value error if `momentum` values are negative:
    if momentum.min() < 0:
       raise ValueError('Momentum magnitude values must be nonnegative.')
    
    # Define a list of arrays containing all possible four particle
    # (left- and right-) handedness configurations:
    h_ll = handedness_config(4, [2, 3], [-1, -1])
    h_lr = handedness_config(4, [2, 3], [-1, 1])
    h_rl = handedness_config(4, [2, 3], [1, -1])
    h_rr = handedness_config(4, [2, 3], [1, 1])
    h_list = [h_ll, h_lr, h_rl, h_rr]
    
    # Initialise int of number of True inputs:
    num_output = 0

    # Initialise dictionary of output values:
    output_dictionary = {}
    
    # Initialise output values arrays if given not False:
    # Update the "number of True inputs" parameter:
    if dp == True:

        num_output += 1

        dp_array = np.array([[[0.0 for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])
        
    if dcs == True:

        num_output += 1

        dcs_array = np.array([[[0.0 for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if c == True:

        num_output += 1

        conc_array = np.array([[[0.0 for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if stokes == True:

        num_output += 15

        s_array = np.array([[[[0.0 for _ in range(15)] \
            for _ in range(len(phi))] for _ in range(len(theta))] \
            for _ in range(len(momentum))])

    if deg_pol == True:

        num_output += 1

        pol_array = np.array([[[0.0 for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])
    
    if amplitudes == True:

        num_output += 16

        amplitudes_array = np.array([[[[0.0 + 0.0j for _ in range(16)] \
            for _ in range(len(phi))] for _ in range(len(theta))] \
            for _ in range(len(momentum))])
        
    if out_state == True:

        num_output += 1

        out_state_array = np.empty((len(momentum),len(theta),len(phi)), \
                                    dtype=QuantumState)

    #
    energy_e = np.sqrt(momentum**2 + m_e**2)
    energy_mu = np.sqrt(momentum**2 + m_mu**2)
    
    # Initialise incoming four-momentum lists:
    p_in_1 = []    # Incoming electron four-momentum
    p_in_2 = []
    
    # Loop over the size of the list of momenta:
    for p_index in range(len(momentum)):

        if scattering == 'compton':

            # Append electron and photon four-momenta:
            p_in_1.append(FourVector(energy_e[p_index], \
            momentum[p_index], 0, 0))
            p_in_2.append(FourVector(momentum[p_index], \
            momentum[p_index], np.pi, 0))
        
        if scattering == 'bhabha':

            # Append electron and photon four-momenta:
            p_in_1.append(FourVector(energy_e[p_index], \
            momentum[p_index], 0, 0))
            p_in_2.append(FourVector(energy_e[p_index], \
            momentum[p_index], np.pi, 0))
        
        if scattering == 'moller':

            # Append electron and photon four-momenta:
            p_in_1.append(FourVector(energy_e[p_index], \
            momentum[p_index], 0, 0))
            p_in_2.append(FourVector(energy_e[p_index], \
            momentum[p_index], np.pi, 0))
        
        if scattering == 'electron_positron_annihilation':

            # Append electron and photon four-momenta:
            p_in_1.append(FourVector(energy_e[p_index], \
            momentum[p_index], 0, 0))
            p_in_2.append(FourVector(energy_e[p_index], \
            momentum[p_index], np.pi, 0))

        if scattering == 'electron_muon':

            # Append electron and photon four-momenta:
            p_in_1.append(FourVector(energy_e[p_index], \
            momentum[p_index], 0, 0))
            p_in_2.append(FourVector(energy_mu[p_index], \
            momentum[p_index], np.pi, 0))
        
        if scattering == 'electron_positron_to_muon_antimuon':

            # Append electron and photon four-momenta:
            p_in_1.append(FourVector(energy_e[p_index], \
            momentum[p_index], 0, 0))
            p_in_2.append(FourVector(energy_e[p_index], \
            momentum[p_index], np.pi, 0))

    # Define a list of 12 empty lists:
    output_data = empty_lists(num_output)

    # Loop over the size of the list of momenta:
    for momentum_index in range(len(momentum)):
    
        # Print percentage of progress:
        progress(momentum_index, len(momentum))

        # Loop over the size of the list of theta angles:
        for theta_index in range(len(theta)):

            # Loop over the size of the list of phi angles:
            for phi_index in range(len(phi)):

                if scattering == 'compton':

                    # Raise ValueError for divergent momentum values: 
                    if momentum.min() < 1e-16:
                        raise ValueError("Compton scattering in the centre-" \
                        "of-mass frame must be for particles of |p| <" \
                        " 1e-16 MeV.")

                    # Array of the incoming electron's energy values 
                    energy = np.sqrt(momentum**2 + m_e**2)

                    p_fermion_in = \
                        p_in_1[momentum_index]
                    
                    p_photon_in = \
                        p_in_2[momentum_index]
                    
                    p_fermion_out = FourVector(energy[momentum_index], \
                        momentum[momentum_index], theta[theta_index], \
                        phi[phi_index])
                    
                    p_photon_out = FourVector(momentum[momentum_index], \
                        momentum[momentum_index], np.pi - theta[theta_index], \
                        phi[phi_index] + np.pi)
                    
                    # Define empty array where a 4x4 scattering matrix is
                    # to be appended:
                    M_matrix = []
                    
                    # Loop over all arrays in h_list:
                    for h_list_index in range(len(h_list)):
                        
                        # Define an empty scattering matrix row term for fixed 
                        # final particle helicity and polarization:
                        M_matrix_row = []
                        
                        # Loop over all helicity configurations in a single 
                        # array:
                        for h_array_index in range(len(h_list[h_list_index])):
                            
                            # Four-momenta of virtual fermion for
                            # s-channel Compton scattering:
                            q_virtual_s = p_fermion_in + p_photon_in

                            # u-channel Compton scattering:
                            q_virtual_u = p_fermion_in - p_photon_out

                            # Define all four RealParticle objects:
                            electron_in = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][0], \
                                p_fermion_in, 'in')
                            
                            photon_in = RealParticle.photon( \
                                h_list[h_list_index][h_array_index][1], \
                                    p_photon_in, 'in')
                            
                            electron_out = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][2], \
                                    p_fermion_out, 'out')
                            
                            photon_out = RealParticle.photon( \
                                h_list[h_list_index][h_array_index][3], \
                                    p_photon_out, 'out')

                            # Define VirtualParticle objects for s and u 
                            # channel:
                            electron_virtual_s = VirtualParticle.electron( \
                                q_virtual_s)
                            
                            electron_virtual_u = VirtualParticle.electron( \
                                q_virtual_u)

                            # Define Dirac spinors in the helicity basis:
                            u_electron_in = electron_in.polarization.bispinor
            
                            u_electron_out = electron_out.polarization.bispinor
                            
                            # Define photon polarization four-vectors:
                            e_photon_in = -1j * e * slashed( \
                                photon_in.polarization.vector)
                            
                            e_photon_out = -1j * e * slashed( \
                                photon_out.polarization.vector)

                            # Define the propagator terms for s and u channels:
                            g_s = electron_virtual_s.propagator
                            g_u = electron_virtual_u.propagator

                            # Total amplitude:
                            L1 = u_electron_out.dot(e_photon_out)
                            R1 = e_photon_in.dot(u_electron_in)
                            M1 = np.dot(L1, np.dot(g_s, R1))
                
                            L2 = u_electron_out.dot(e_photon_in)
                            R2 = e_photon_out.dot(u_electron_in)
                            M2 = np.dot(L2, np.dot(g_u, R2))
                            
                            # Calculate total scattering amplitude
                            M_matrix_term = M1 + M2
                            
                            # Append to M_matrix_row
                            M_matrix_row.append(M_matrix_term)

                        # Append to the array of all amplitudes:
                        M_matrix.append(M_matrix_row)

                    # Find the scattered electron-positron state:
                    out_state_var = QuantumState.out_state( \
                                    in_state, M_matrix)

                elif scattering == 'bhabha':

                    # Raise ValueError for IR-divergent momentum values: 
                    if momentum.min() < 1e-16:
                        raise ValueError("Bhabha scattering amplitudes " \
                            "are divergent for momentum < 1e-17 MeV.")
                    
                    # Raise ValueError for collinear-divergent theta values: 
                    if theta.min() < 1e-16:
                        raise ValueError("Bhabha scattering amplitudes " \
                            "are divergent for theta < 1e-17 rad.")

                    # Array of the incoming electron's energy values 
                    energy = np.sqrt(momentum**2 + m_e**2)

                    p_electron_in = \
                        p_in_1[momentum_index]
                    
                    p_positron_in = \
                        p_in_2[momentum_index]
                    
                    p_electron_out = FourVector(energy[momentum_index], \
                        momentum[momentum_index], theta[theta_index], \
                        phi[phi_index])
                    
                    p_positron_out = FourVector(energy[momentum_index], \
                        momentum[momentum_index], np.pi - theta[theta_index],\
                        phi[phi_index] + np.pi)
                    
                    # Define empty array where a 4x4 scattering matrix is
                    # to be appended:
                    M_matrix = []
                    
                    # Loop over all arrays in h_list:
                    for h_list_index in range(len(h_list)):
                        
                        # Define an empty scattering matrix row term for fixed 
                        # final particle helicity and polarization:
                        M_matrix_row = []
                        
                        # Loop over all helicity configurations in a single 
                        # array:
                        for h_array_index in range(len(h_list[h_list_index])):
                            
                            # Four-momenta of virtual fermion for
                            # s-channel Bhabha scattering:
                            q_virtual_s = p_electron_in + p_positron_in

                            # t-channel Bhabha scattering:
                            q_virtual_t = p_electron_in - p_electron_out

                            # Define all four RealParticle objects:
                            electron_in = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][0], \
                                p_electron_in, 'in')
                            
                            positron_in = RealParticle.positron( \
                                h_list[h_list_index][h_array_index][1], \
                                    p_positron_in, 'in')
                            
                            electron_out = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][2], \
                                    p_electron_out, 'out')
                            
                            positron_out = RealParticle.positron( \
                                h_list[h_list_index][h_array_index][3], \
                                    p_positron_out, 'out')

                            # Define VirtualParticle objects for s and t 
                            # channel:
                            photon_virtual_s = VirtualParticle.photon( \
                                q_virtual_s)
                            
                            photon_virtual_t = VirtualParticle.photon( \
                                q_virtual_t)

                            # Define Dirac spinors in the helicity basis:
                            u_electron_in = electron_in.polarization.bispinor
            
                            u_electron_out = electron_out.polarization.bispinor

                            v_positron_in = positron_in.polarization.bispinor
            
                            v_positron_out = positron_out.polarization.bispinor
                            
                            # Define the propagator terms for s and t channels:
                            g_s = photon_virtual_s.propagator
                            g_u = photon_virtual_t.propagator

                            # s-channel amplitude
                            Js_i = -1j * e * v_positron_in.dot(GAMMA).\
                            dot(u_electron_in)
                            Js_o = -1j * e * u_electron_out.dot(GAMMA).\
                                dot(v_positron_out)
                            M_s = -1j * lorentzian_product(Js_i, Js_o) \
                                / lorentzian_product(q_virtual_s, q_virtual_s)
                    
                            # t-channel amplitude
                            Jt_i = -1j * e * u_electron_out.dot(GAMMA).\
                                dot(u_electron_in)
                            Jt_o = -1j * e * v_positron_in.dot(GAMMA).\
                                dot(v_positron_out)
                            M_t = -1j * lorentzian_product(Jt_i, Jt_o) \
                                / lorentzian_product(q_virtual_t, q_virtual_t)
                    
                            # Total amplitude
                            M_matrix_term = -M_t + M_s
                            
                            # Append to M_matrix_row
                            M_matrix_row.append(M_matrix_term)

                        # Append to the array of all amplitudes:
                        M_matrix.append(M_matrix_row)
                    
                    # Find the scattered electron-positron state:
                    out_state_var = QuantumState.out_state( \
                                    in_state, M_matrix)
                
                elif scattering == 'moller':

                    # Raise ValueError for divergent momentum values: 
                    if momentum.min() < 1e-16:
                        raise ValueError("Moller scattering amplitudes are " \
                        "divergent for |p| < 1e-16 MeV.")
                    
                    # Raise ValueError for divergent theta values: 
                    if theta.min() < 1e-16:
                        raise ValueError("Moller scattering amplitudes are " \
                        "divergent for theta < 1e-16 rad.")
                    
                    # Raise ValueError for divergent theta values: 
                    if theta.max() > np.pi - 1e-16:
                        raise ValueError("Moller scattering amplitudes are " \
                        "divergent for theta < np.pi - 1e-16 rad.")

                    # Array of electron's energy values 
                    energy_e = np.sqrt(momentum**2 + m_e**2)
                    
                    p_electron1_in = \
                        p_in_1[momentum_index]
                    
                    p_electron2_in = \
                        p_in_2[momentum_index]
                    
                    p_electron1_out = FourVector(energy_e[momentum_index], \
                        momentum[momentum_index], theta[theta_index], \
                        phi[phi_index])
                    
                    p_electron2_out = FourVector(energy_e[momentum_index], \
                        momentum[momentum_index], np.pi - theta[theta_index], \
                        phi[phi_index] + np.pi)
                    
                    # Define empty array where a 4x4 scattering matrix is
                    # to be appended:
                    M_matrix = []
                    
                    # Loop over all arrays in h_list:
                    for h_list_index in range(len(h_list)):
                        
                        # Define an empty scattering matrix row term for fixed 
                        # final particle helicity and polarization:
                        M_matrix_row = []
                        
                        # Loop over all helicity configurations in a single 
                        # array:
                        for h_array_index in range(len(h_list[h_list_index])):
                            
                            # Four-momenta of virtual photons:
                            q_virtual_t = p_electron1_in - p_electron1_out
                            q_virtual_u = p_electron1_in - p_electron2_out

                            # Define all four RealParticle objects:
                            electron1_in = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][0], \
                                p_electron1_in, 'in')
                            
                            electron2_in = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][1], \
                                    p_electron2_in, 'in')
                            
                            electron1_out = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][2], \
                                    p_electron1_out, 'out')
                            
                            electron2_out = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][3], \
                                    p_electron2_out, 'out')
                        
                            # Define Dirac spinors in the helicity basis:
                            u_electron1_in = \
                                electron1_in.polarization.bispinor
            
                            u_electron1_out = \
                            electron1_out.polarization.bispinor

                            u_electron2_in = \
                                electron2_in.polarization.bispinor
            
                            u_electron2_out = \
                                electron2_out.polarization.bispinor
                            
                            # t-channel amplitude
                            J_t1 = -1j * e * u_electron1_out.dot(GAMMA).\
                                dot(u_electron1_in)
                            
                            J_t2 = -1j * e * u_electron2_out.dot(GAMMA).\
                                dot(u_electron2_in)
                            
                            M_t = -1j * lorentzian_product(J_t1, J_t2) \
                                / lorentzian_product(q_virtual_t, q_virtual_t)
                            
                            # t-channel amplitude
                            J_u1 = -1j * e * u_electron2_out.dot(GAMMA).\
                                dot(u_electron1_in)
                            
                            J_u2 = -1j * e * u_electron1_out.dot(GAMMA).\
                                dot(u_electron2_in)
                            
                            M_u = -1j * lorentzian_product(J_u1, J_u2) \
                                / lorentzian_product(q_virtual_u, q_virtual_u)
                        
                            # Total amplitude
                            M_matrix_term = M_t - M_u
                            
                            # Append to M_matrix_row
                            M_matrix_row.append(M_matrix_term)

                        # Append to the array of all amplitudes:
                        M_matrix.append(M_matrix_row)

                    # Find the scattered electron-positron state:
                    out_state_var = QuantumState.out_state( \
                                    in_state, M_matrix)

                elif scattering == 'electron_positron_annihilation':

                    # Array of the incoming electron's energy values 
                    energy = np.sqrt(momentum**2 + m_e**2)
                    
                    p_electron_in = \
                        p_in_1[momentum_index]
                    
                    p_positron_in = \
                        p_in_2[momentum_index]
                    
                    p_photon1_out = FourVector(energy[momentum_index], \
                        energy[momentum_index], theta[theta_index], \
                        phi[phi_index])
                    
                    p_photon2_out = FourVector(energy[momentum_index], \
                        energy[momentum_index], np.pi - theta[theta_index], \
                        phi[phi_index] + np.pi)
                    
                    # Define empty array where a 4x4 scattering matrix is
                    # to be appended:
                    M_matrix = []
                    
                    # Loop over all arrays in h_list:
                    for h_list_index in range(len(h_list)):
                        
                        # Define an empty scattering matrix row term for fixed 
                        # final particle helicity and polarization:
                        M_matrix_row = []
                        
                        # Loop over all helicity configurations in a single 
                        # array:
                        for h_array_index in range(len(h_list[h_list_index])):
                            
                            # Four-momenta of virtual fermion for
                            # t-channel scattering:
                            q_virtual_t = p_electron_in - p_photon1_out

                            # u-channel scattering:
                            q_virtual_u = p_electron_in - p_photon2_out

                            # Define all four RealParticle objects:
                            electron_in = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][0], \
                                p_electron_in, 'in')
                            
                            positron_in = RealParticle.positron( \
                                h_list[h_list_index][h_array_index][1], \
                                    p_positron_in, 'in')
                            
                            photon1_out = RealParticle.photon( \
                                h_list[h_list_index][h_array_index][2], \
                                    p_photon1_out, 'out')
                            
                            photon2_out = RealParticle.photon( \
                                h_list[h_list_index][h_array_index][3], \
                                    p_photon2_out, 'out')

                            # Define VirtualParticle objects for s and u channel:
                            electron_virtual_t = VirtualParticle.electron( \
                                q_virtual_t)
                            
                            electron_virtual_u = VirtualParticle.electron( \
                                q_virtual_u)

                            # Define Dirac spinors in the helicity basis:
                            u_electron_in = electron_in.polarization.bispinor
            
                            v_positron_in = positron_in.polarization.bispinor
                            
                            # Define photon polarization four-vectors:
                            e_photon1_out = -1j * e * slashed( \
                                photon1_out.polarization.vector)
                            
                            e_photon2_out = -1j * e * slashed( \
                                photon2_out.polarization.vector)

                            # Define the propagator terms for t and u channels:
                            g_t = electron_virtual_t.propagator
                            g_u = electron_virtual_u.propagator

                            # Calculate total scattering amplitude
                            M_matrix_term = v_positron_in.dot(e_photon2_out).\
                            dot(g_t).dot(e_photon1_out).dot(u_electron_in) + \
                            v_positron_in.dot(e_photon1_out).dot(g_u).\
                            dot(e_photon2_out).dot(u_electron_in)
                                
                            # Append to M_matrix_row
                            M_matrix_row.append(M_matrix_term)

                        # Append to the array of all amplitudes:
                        M_matrix.append(M_matrix_row)
                    
                    # Find the scattered electron-positron state:
                    out_state_var = QuantumState.out_state( \
                                    in_state, M_matrix)

                elif scattering == 'electron_muon':

                    # Raise ValueError for divergent momentum values: 
                    if momentum.min() < 1e-16:
                        raise ValueError("Electron-muon scattering amplitudes"\
                        "are divergent for |p| < 1e-16 MeV.")
                    
                    # Raise ValueError for divergent theta values: 
                    if theta.min() < 1e-16:
                        raise ValueError("Electron-muon scattering amplitudes"\
                        "are divergent for theta < 1e-16 rad.")

                    # Array of electron's and muon's energy values 
                    energy_e = np.sqrt(momentum**2 + m_e**2)
                    energy_mu = np.sqrt(momentum**2 + m_mu**2)
                    
                    p_electron_in = \
                        p_in_1[momentum_index]
                    
                    p_muon_in = \
                        p_in_2[momentum_index]
                    
                    p_electron_out = FourVector(energy_e[momentum_index], \
                        momentum[momentum_index], theta[theta_index], \
                        phi[phi_index])
                    
                    p_muon_out = FourVector(energy_mu[momentum_index], \
                        momentum[momentum_index], np.pi - theta[theta_index], \
                        phi[phi_index] + np.pi)
                    
                    # Define empty array where a 4x4 scattering matrix is
                    # to be appended:
                    M_matrix = []
                    
                    # Loop over all arrays in h_list:
                    for h_list_index in range(len(h_list)):
                        
                        # Define an empty scattering matrix row term for fixed 
                        # final particle helicity and polarization:
                        M_matrix_row = []
                        
                        # Loop over all helicity configurations in a single 
                        # array:
                        for h_array_index in range(len(h_list[h_list_index])):
                            
                            # Four-momentum of virtual photon:
                            q_virtual_t = p_electron_in - p_electron_out

                            # Define all four RealParticle objects:
                            electron_in = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][0], \
                                p_electron_in, 'in')
                            
                            muon_in = RealParticle.muon( \
                                h_list[h_list_index][h_array_index][1], \
                                    p_muon_in, 'in')
                            
                            electron_out = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][2], \
                                    p_electron_out, 'out')
                            
                            muon_out = RealParticle.muon( \
                                h_list[h_list_index][h_array_index][3], \
                                    p_muon_out, 'out')

                            # Define VirtualParticle object:
                            photon_virtual_t = VirtualParticle.photon( \
                                q_virtual_t)
                        
                            # Define Dirac spinors in the helicity basis:
                            u_electron_in = electron_in.polarization.bispinor
            
                            u_electron_out = electron_out.polarization.bispinor

                            u_muon_in = muon_in.polarization.bispinor
            
                            u_muon_out = muon_out.polarization.bispinor
                            
                            # Define the propagator terms for s and t channels:
                            g_t = photon_virtual_t.propagator

                            # t-channel amplitude
                            J_e = -1j * e * u_electron_out.dot(GAMMA).\
                                dot(u_electron_in)
                            J_mu = -1j * e * u_muon_out.dot(GAMMA).\
                                dot(u_muon_in)
                            M_matrix_term = -1j * \
                                lorentzian_product(J_e, J_mu) \
                                / lorentzian_product(q_virtual_t, q_virtual_t)
                            
                            # Append to M_matrix_row
                            M_matrix_row.append(M_matrix_term)

                        # Append to the array of all amplitudes:
                        M_matrix.append(M_matrix_row)

                    # Find the scattered electron-positron state:
                    out_state_var = QuantumState.out_state( \
                                    in_state, M_matrix)

                elif scattering == 'electron_positron_to_muon_antimuon':
                    
                    # Introduce physical momentum limit error unique to 
                    # electron positron to muon antimuon scattering:
                    if momentum.min() < np.sqrt(MUON_MASS**2-ELECTRON_MASS**2):
                        raise ValueError("Input momentum values" \
                        " must be larger than np.sqrt(MUON_MASS**2"\
                        "-ELECTRON_MASS**2).")

                    # Array of electron's energy values 
                    energy_e = np.sqrt(momentum**2 + m_e**2)

                    # Define incoming and outgoing four-momenta:
                    p_electron_in = \
                        p_in_1[momentum_index]
                    
                    p_positron_in = \
                        p_in_2[momentum_index]
                    
                    p_muon_out = FourVector(energy_e[momentum_index], \
                        np.sqrt(energy_e[momentum_index]**2 - m_mu**2), \
                        theta[theta_index], phi[phi_index])
                    
                    p_antimuon_out = FourVector(energy_e[momentum_index], \
                        np.sqrt(energy_e[momentum_index]**2 - m_mu**2), \
                        np.pi - theta[theta_index], phi[phi_index] + np.pi)
                    
                    # Define empty array where a 4x4 scattering matrix is
                    # to be appended:
                    M_matrix = []
                    
                    # Loop over all arrays in h_list:
                    for h_list_index in range(len(h_list)):
                        
                        # Define an empty scattering matrix row term for fixed 
                        # final particle helicity and polarization:
                        M_matrix_row = []
                        
                        # Loop over all helicity configurations in a single 
                        # array:
                        for h_array_index in range(len(h_list[h_list_index])):
                            
                            # Four-momentum of virtual photon:
                            q_virtual_s = p_electron_in + p_positron_in

                            # Define all four RealParticle objects:
                            electron_in = RealParticle.electron( \
                                h_list[h_list_index][h_array_index][0], \
                                p_electron_in, 'in')
                            
                            positron_in = RealParticle.positron( \
                                h_list[h_list_index][h_array_index][1], \
                                    p_positron_in, 'in')
                            
                            muon_out = RealParticle.muon( \
                                h_list[h_list_index][h_array_index][2], \
                                    p_muon_out, 'out')
                            
                            antimuon_out = RealParticle.antimuon( \
                                h_list[h_list_index][h_array_index][3], \
                                    p_antimuon_out, 'out')

                            # Define VirtualParticle object:
                            photon_virtual_s = VirtualParticle.photon( \
                                q_virtual_s)
                        
                            # Define Dirac spinors in the helicity basis:
                            u_electron_in = electron_in.polarization.bispinor
            
                            v_positron_in = positron_in.polarization.bispinor

                            u_muon_out = muon_out.polarization.bispinor
            
                            v_antimuon_out = antimuon_out.polarization.bispinor
                            
                            # Define the propagator terms for s and t channels:
                            g_s = photon_virtual_s.propagator

                            # t-channel amplitude
                            J_e = -1j * e * v_positron_in.dot(GAMMA).\
                                dot(u_electron_in)
                            J_mu = -1j * e * u_muon_out.dot(GAMMA).\
                                dot(v_antimuon_out)
                            M_matrix_term = -1j * \
                                 lorentzian_product(J_e, J_mu) \
                                / lorentzian_product(q_virtual_s, q_virtual_s)
                            
                            # Append to M_matrix_row
                            M_matrix_row.append(M_matrix_term)

                        # Append to the array of all amplitudes:
                        M_matrix.append(M_matrix_row)

                    # Find the scattered electron-positron state:
                    out_state_var = QuantumState.out_state( \
                                    in_state, M_matrix)
  
                # Initialise empty lists for output dictionary and .pkl 
                # file:
                quantities = []
                keys = []

                # Calculate and append all output quantities if not False:
                if dp == True:

                    dp_array[momentum_index,theta_index,phi_index] = \
                        differential_probability(out_state_var, projection)
                    
                if dcs == True:

                    dcs_array[momentum_index,theta_index,phi_index] = \
                        diff_cross_section(p_in_1, p_in_2, out_state_var, \
                                           projection=None)

                if c == True:

                    conc_array[momentum_index,theta_index,phi_index] \
                        = concurrence(out_state_var)

                if deg_pol == True:
                    
                    pol_array[momentum_index,theta_index,phi_index] = \
                        degree_polarization(out_state_var)

                if stokes == True:

                    for s_k in [1,2,3]:
                            s_array[momentum_index,theta_index,phi_index, \
                                    s_k - 1] \
                            = stokes_parameter(out_state_var, [0, s_k])

                    for s_i in [1,2,3]:
                        for s_j in range(4):
                            s_array[momentum_index,theta_index,phi_index, \
                                    3*s_i + s_j] \
                            = stokes_parameter(out_state_var, [s_i, s_j])
                    
                if amplitudes == True:

                    for a_i in range(4):
                        for a_j in range(4):
                            amplitudes_array[momentum_index,theta_index, \
                            phi_index, 4*a_i + a_j] = M_matrix[a_i][a_j]
                        
                if out_state == True:
                    
                    out_state_array[momentum_index,theta_index,phi_index] \
                        = out_state_var
    
    # Append all output quantities:
    if dp == True:

        quantities.append(dp_array)
        keys.append('dp')
        output_dictionary['dp'] = dp_array.squeeze()

    if c == True:

        quantities.append(conc_array)
        keys.append('c')
        output_dictionary['c'] = conc_array.squeeze()

    if deg_pol == True:

        quantities.append(pol_array)
        keys.append('deg_pol')
        output_dictionary['deg_pol'] = pol_array.squeeze()

    if stokes == True:

        stokes_labels = [
            's01', 's02', 's03',
            's10', 's11', 's12', 's13',
            's20', 's21', 's22', 's23',
            's30', 's31', 's32', 's33'
        ]

        keys.extend(stokes_labels)

        for stok_i, stokes_label in enumerate(stokes_labels):

            quantities.extend([s_array[:,:,:,stok_i]])

            output_dictionary[stokes_label]=s_array[:, :, :, stok_i].squeeze()

    if amplitudes == True:

        hel_str_list = ['l', 'r']
        hel_index = {'l': 0, 'r': 1} 

        for amp_i in range(16):
            quantities.extend([amplitudes_array[:,:,:,amp_i]])

        for a in hel_str_list:
            for b in hel_str_list:
                for c in hel_str_list:
                    for d in hel_str_list:

                        keys.extend([f'{a}{b}_to_{c}{d}'])

                        output_dictionary[f'{a}{b}_to_{c}{d}'] = \
                        amplitudes_array[:,:,:,8 * hel_index[a] + \
                        4 * hel_index[b] + 2 * hel_index[c] + \
                        hel_index[d]].squeeze()


    if out_state == True:

        quantities.append(out_state_array)
        keys.append('out_state')
        output_dictionary['out_state'] = out_state_array.squeeze()

    # Append all calculated outputs to output_data array:
    for data_index in range(len(output_data)):
        output_data[data_index].append(quantities[data_index])
    
    # Save dictionary as a .pkl file:
    if filename is not None:
        if isinstance(filename, str):
            save_data(f'{filename}', keys, quantities)
        else: 
            raise TypeError(f"Expected 'filename' to be a string"\
                        + f", but got {type(filename).__name__}.")
    
    return output_dictionary 
