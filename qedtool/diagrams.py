import numpy as np
from relativity import *
from qed import *
from quoptics import *
from quoptics import QuantumState


# Import electron charge value:
e = COUPLING
    
# Import electron and muon mass
m_e = ELECTRON_MASS
m_mu = MUON_MASS


def compton(in_state, momentum, theta, phi=None, filename=None, 
            projection_state=None, diff_probability=False, 
            state_concurrence=False, stokes=False, deg_polarization=False):
    """
    Calculates differential probability, output state concurrence, Stokes 
    parameters, and degree of polarization for a tree level Compton scattering
    process in the centre-of-mass frame. 

    Parameters
    ----------
    in_state : QuantumState
        Quantum state of incoming electron and photon particles.
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
    projection_state : QuantumState 
        Optional input of the state onto which the outgoing (scattered) 
        electron-photon state is projected, such that differential proobability
        values are adjusted accordingly. `False` by default.
    filename : str 
        If `filename` is `str` the output data dictionary is saved as a 
        Pickle file, with the string as a title. `None` by default.
    diff_probability : bool 
        If `diff_probability` is `True` an array of values of the spin-averaged 
        sum of |M|^2 (the squared magnitude of amplitudes) is output. The array 
        runs over all input `momentum`, `theta`, and `phi` values. `False` by 
        default.
    state_concurrence : bool 
        If `state_concurrence` is `True` an array of concurrence values of the 
        scattered state is output. The array runs over all input `momentum`, 
        `theta`, and `phi` values. `False` by default.
    stokes : bool 
        If `state_concurrence` is `True` an array of the Stokes parameters of 
        the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    deg_polarization : bool 
        If `state_concurrence` is `True` an array of the degree of polarization 
        values of the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    n_units : int
        Redefines all global variables, such that all calculations are done in 
        XeV units, where the placeholder X stands for the unit prefix 
        corresponding to an order of magnitude of n according to the standard 
        convention.

    Returns
    -------
    output_dictionary : dict
        Dictionary of `diff_probability`, `state_concurrence`, `stokes`, 
        `deg_polarization` arrays.
    ---------------------------------------------------------------------------
    
    """

    # Raise a type error if in_state and/or projection_state are not of 
    # `QuantumState` type:
    if not isinstance(in_state, QuantumState):
        raise TypeError("Expected 'in_state' to be of type `QuantumState`," \
                        + f" but got {type(in_state).__name__}.")
    
    if projection_state is not None:
        if not isinstance(projection_state, QuantumState):
            raise TypeError("Expected 'projection_state' to be of type"\
            + f" `QuantumState`, but got {type(projection_state).__name__}.")
        
        if np.shape(projection_state.rho) != (4, 4):
            raise TypeError("Expected 'projection_state' to be"\
            + f" of shape (4,4), but got {np.shape(projection_state.rho)}.")
    
    if np.shape(in_state.rho) != (4, 4):
        raise TypeError("Expected 'in_state' to be of"\
                        + f" shape (4,4), but got {np.shape(in_state.rho)}.")


    # Raise a type error if `momentum`, `theta`, and `phi` are not of lists
    # or NumPy arrays:
    if momentum is not None:
        if not isinstance(momentum, (list, np.ndarray)):
            raise TypeError("Expected 'energy' to be a list or a NumPy array,"\
                        + f" but got {type(momentum).__name__}.")
        
    if theta is not None:
        if not isinstance(theta, (list, np.ndarray)):
            raise TypeError("Expected 'theta' to be a list or a NumPy array,"\
                        + f" but got {type(theta).__name__}.")
    
    if phi is not None:
        if not isinstance(phi, (list, np.ndarray)):
            raise TypeError(f"Expected 'phi' to be a list or a NumPy array"\
                        + f", but got {type(phi).__name__}.")
    elif phi == None:
        phi = [0]

    # Convert input array_like objects to NumPy arrays:
    momentum = np.array(momentum)
    theta = np.array(theta)
    phi = np.array(phi)

    # Array of the incoming electron's energy values 
    energy = np.sqrt(momentum**2 + m_e**2)
    
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
    
    # Initialise output values array if given not False:
    # Update the "number of True inputs" parameter:
    if diff_probability != False:

        num_output += 1

        dW_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if state_concurrence != False:

        num_output += 1

        conc_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if stokes != False:

        num_output += 9

        s_array = np.array([[[[0 + 0j for _ in range(9)] \
            for _ in range(len(phi))] for _ in range(len(theta))] \
            for _ in range(len(momentum))])

    if deg_polarization != False:

        num_output += 1

        pol_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    # Define a list of 12 empty lists:
    output_data = empty_lists(num_output)

    # Loop over the size of the list of momenta:
    for momentum_index in range(len(momentum)):

        # Loop over the size of the list of theta angles:
        for theta_index in range(len(theta)):

            # Loop over the size of the list of phi angles:
            for phi_index in range(len(phi)):

                p_fermion_in = FourVector(energy[momentum_index], \
                    momentum[momentum_index], 0, 0)
                
                p_photon_in = FourVector(momentum[momentum_index], \
                    momentum[momentum_index], np.pi, 0)
                
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
                    
                    # Loop over all helicity configurations in a single array :
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

                        # Define VirtualParticle objects for s and u channel:
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

                # Find the scattered electron-photon state:
                out_state = QuantumState.out_state(in_state, M_matrix)
                
                # Initialise empty lists for output ictionary and .pkl file:
                quantities = []
                keys = []

                # Calculate and append all output quantities if not False:
                if diff_probability != False:

                    dW_array[momentum_index,theta_index,phi_index] = \
                        differential_probability(out_state, projection_state)

                if state_concurrence != False:

                    conc_array[momentum_index,theta_index,phi_index] \
                        = concurrence(out_state)

                if deg_polarization != False:
                    
                    pol_array[momentum_index,theta_index,phi_index] = \
                        deg_of_polarization(out_state)

                if stokes != False:

                    s_array[momentum_index,theta_index,phi_index,0] \
                          = stokes_parameter(out_state, [1, 1])
                    s_array[momentum_index,theta_index,phi_index,1] \
                          = stokes_parameter(out_state, [1, 2])
                    s_array[momentum_index,theta_index,phi_index,2] \
                          = stokes_parameter(out_state, [1, 3])
                    s_array[momentum_index,theta_index,phi_index,3] \
                          = stokes_parameter(out_state, [2, 1])
                    s_array[momentum_index,theta_index,phi_index,4] \
                          = stokes_parameter(out_state, [2, 2])
                    s_array[momentum_index,theta_index,phi_index,5] \
                          = stokes_parameter(out_state, [2, 3])
                    s_array[momentum_index,theta_index,phi_index,6] \
                          = stokes_parameter(out_state, [3, 1])
                    s_array[momentum_index,theta_index,phi_index,7] \
                          = stokes_parameter(out_state, [3, 2])
                    s_array[momentum_index,theta_index,phi_index,8] \
                          = stokes_parameter(out_state, [3, 3])
    
    # Append all output quantities:
    if diff_probability != False:

        quantities.append(dW_array)
        keys.append('diff_probability')
        output_dictionary['diff_probability'] = dW_array.squeeze()

    if state_concurrence != False:

        quantities.append(conc_array)
        keys.append('state_concurrence')
        output_dictionary['state_concurrence'] = conc_array.squeeze()

    if deg_polarization != False:

        quantities.append(pol_array)
        keys.append('deg_polarization')
        output_dictionary['deg_polarization'] = pol_array.squeeze()

    if stokes != False:

        quantities.extend([s_array[:,:,:,0], s_array[:,:,:,1], \
            s_array[:,:,:,2], s_array[:,:,:,3], s_array[:,:,:,4], \
            s_array[:,:,:,5], s_array[:,:,:,6], s_array[:,:,:,7], \
            s_array[:,:,:,8]])
                    
        keys.extend(['s11', 's12', 's13', 's21', 's22', 's23', \
                                 's31', 's32', 's33'])
        
        output_dictionary['s11'] = s_array[:,:,:,0].squeeze()
        output_dictionary['s12'] = s_array[:,:,:,1].squeeze()
        output_dictionary['s13'] = s_array[:,:,:,2].squeeze()
        output_dictionary['s21'] = s_array[:,:,:,3].squeeze()
        output_dictionary['s22'] = s_array[:,:,:,4].squeeze()
        output_dictionary['s23'] = s_array[:,:,:,5].squeeze()
        output_dictionary['s31'] = s_array[:,:,:,6].squeeze()
        output_dictionary['s32'] = s_array[:,:,:,7].squeeze()
        output_dictionary['s33'] = s_array[:,:,:,8].squeeze()
    
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


def bhabha(in_state, momentum, theta, phi=None, projection_state=None, 
           filename=None, diff_probability=False, state_concurrence=False, 
           stokes=False, deg_polarization=False):
    """
    Calculates differential probability, output state concurrence, Stokes 
    parameters, and degree of polarization for a tree level Bhabha scattering
    process in the centre-of-mass frame. 

    Parameters
    ----------
    in_state : QuantumState
        Quantum state of incoming electron and positron particles.
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
    projection_state : QuantumState 
        Optional input of the state onto which the outgoing (scattered) 
        electron-positron state is projected, such that differential 
        proobability values are adjusted accordingly. `False` by default.
    filename : str 
        If `filename` is `str` the output data dictionary is saved as a 
        Pickle file, with the string as a title. `None` by default.
    diff_probability : bool 
        If `diff_probability` is `True` an array of values of the spin-averaged 
        sum of |M|^2 (the squared magnitude of amplitudes) is output. The array 
        runs over all input `momentum`, `theta`, and `phi` values. `False` by 
        default.
    state_concurrence : bool 
        If `state_concurrence` is `True` an array of concurrence values of the 
        scattered state is output. The array runs over all input `momentum`, 
        `theta`, and `phi` values. `False` by default.
    stokes : bool 
        If `state_concurrence` is `True` an array of the Stokes parameters of 
        the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    deg_polarization : bool 
        If `state_concurrence` is `True` an array of the degree of polarization 
        values of the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    n_units : int
        Redefines all global variables, such that all calculations are done in 
        XeV units, where the placeholder X stands for the unit prefix 
        corresponding to an order of magnitude of n according to the standard 
        convention.

    Returns
    -------
    output_dictionary : dict
        Dictionary of `diff_probability`, `state_concurrence`, `stokes`, 
        `deg_polarization` arrays.
    ---------------------------------------------------------------------------
    
    """

    # Raise a type error if in_state and/or projection_state are not of 
    # `QuantumState` type:
    if not isinstance(in_state, QuantumState):
        raise TypeError("Expected 'in_state' to be of type `QuantumState`," \
                        + f" but got {type(in_state).__name__}.")
    
    if projection_state is not None:
        if not isinstance(projection_state, QuantumState):
            raise TypeError("Expected 'in_state' to be of type `QuantumState`"\
                        + f", but got {type(projection_state).__name__}.")
        
        if np.shape(projection_state.rho) != (4, 4):
            raise TypeError("Expected 'in_state' to be a two-particle state"\
            + f" of shape (4,4), but got {np.shape(projection_state.rho)}.")
    
    if np.shape(in_state.rho) != (4, 4):
        raise TypeError("Expected 'in_state' to be a two-particle state of"\
                        + f" shape (4,4), but got {np.shape(in_state.rho)}.")


    # Raise a type error if `momentum`, `theta`, and `phi` are not of lists
    # or NumPy arrays:
    if momentum is not None:
        if not isinstance(momentum, (list, np.ndarray)):
            raise TypeError("Expected 'energy' to be a list or a NumPy array,"\
                        + f" but got {type(momentum).__name__}.")
        
    if theta is not None:
        if not isinstance(theta, (list, np.ndarray)):
            raise TypeError("Expected 'energy' to be a list or a NumPy array,"\
                        + f" but got {type(theta).__name__}.")
    
    if phi is not None:
        if not isinstance(phi, (list, np.ndarray)):
            raise TypeError(f"Expected 'energy' to be a list or a NumPy array"\
                        + f", but got {type(phi).__name__}.")
    elif phi == None:
        phi = [0]

    # Convert input array_like objects to NumPy arrays:
    momentum = np.array(momentum)
    theta = np.array(theta)
    phi = np.array(phi)

    # Array of the incoming electron's energy values 
    energy = np.sqrt(momentum**2 + m_e**2)
    
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
    
    # Initialise output values array if given not False:
    # Update the "number of True inputs" parameter:
    if diff_probability != False:

        num_output += 1

        dW_array = np.array([[[1e-20 + 1e-20j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if state_concurrence != False:

        num_output += 1

        conc_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if stokes != False:

        num_output += 9

        s_array = np.array([[[[0 + 0j for _ in range(9)] \
            for _ in range(len(phi))] for _ in range(len(theta))] \
            for _ in range(len(momentum))])

    if deg_polarization != False:

        num_output += 1

        pol_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    # Define a list of 12 empty lists:
    output_data = empty_lists(num_output)

    # Loop over the size of the list of momenta:
    for momentum_index in range(len(momentum)):

        # Loop over the size of the list of theta angles:
        for theta_index in range(len(theta)):

            # Loop over the size of the list of phi angles:
            for phi_index in range(len(phi)):

                p_electron_in = FourVector(energy[momentum_index], \
                    momentum[momentum_index], 0, 0)
                
                p_positron_in = FourVector(energy[momentum_index], \
                    momentum[momentum_index], np.pi, 0)
                
                p_electron_out = FourVector(energy[momentum_index], \
                    momentum[momentum_index], theta[theta_index], \
                    phi[phi_index])
                
                p_positron_out = FourVector(energy[momentum_index], \
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
                    
                    # Loop over all helicity configurations in a single array :
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

                        # Define VirtualParticle objects for s and t channel:
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
                out_state = QuantumState.out_state(in_state, M_matrix)
                
                # Initialise empty lists for output ictionary and .pkl file:
                quantities = []
                keys = []

                # Calculate and append all output quantities if not False:
                if diff_probability != False:

                    dW_array[momentum_index,theta_index,phi_index] = \
                        differential_probability(out_state, projection_state)

                if state_concurrence != False:

                    conc_array[momentum_index,theta_index,phi_index] \
                        = concurrence(out_state)

                if deg_polarization != False:
                    
                    pol_array[momentum_index,theta_index,phi_index] = \
                        deg_of_polarization(out_state)

                if stokes != False:

                    s_array[momentum_index,theta_index,phi_index,0] \
                          = stokes_parameter(out_state, [1, 1])
                    s_array[momentum_index,theta_index,phi_index,1] \
                          = stokes_parameter(out_state, [1, 2])
                    s_array[momentum_index,theta_index,phi_index,2] \
                          = stokes_parameter(out_state, [1, 3])
                    s_array[momentum_index,theta_index,phi_index,3] \
                          = stokes_parameter(out_state, [2, 1])
                    s_array[momentum_index,theta_index,phi_index,4] \
                          = stokes_parameter(out_state, [2, 2])
                    s_array[momentum_index,theta_index,phi_index,5] \
                          = stokes_parameter(out_state, [2, 3])
                    s_array[momentum_index,theta_index,phi_index,6] \
                          = stokes_parameter(out_state, [3, 1])
                    s_array[momentum_index,theta_index,phi_index,7] \
                          = stokes_parameter(out_state, [3, 2])
                    s_array[momentum_index,theta_index,phi_index,8] \
                          = stokes_parameter(out_state, [3, 3])
    
    # Append all output quantities:
    if diff_probability != False:

        quantities.append(dW_array)
        keys.append('diff_probability')
        output_dictionary['diff_probability'] = dW_array.squeeze()

    if state_concurrence != False:

        quantities.append(conc_array)
        keys.append('state_concurrence')
        output_dictionary['state_concurrence'] = conc_array.squeeze()

    if deg_polarization != False:

        quantities.append(pol_array)
        keys.append('deg_polarization')
        output_dictionary['deg_polarization'] = pol_array.squeeze()

    if stokes != False:

        quantities.extend([s_array[:,:,:,0], s_array[:,:,:,1], \
            s_array[:,:,:,2], s_array[:,:,:,3], s_array[:,:,:,4], \
            s_array[:,:,:,5], s_array[:,:,:,6], s_array[:,:,:,7], \
            s_array[:,:,:,8]])
                    
        keys.extend(['s11', 's12', 's13', 's21', 's22', 's23', \
                                 's31', 's32', 's33'])
        
        output_dictionary['s11'] = s_array[:,:,:,0].squeeze()
        output_dictionary['s12'] = s_array[:,:,:,1].squeeze()
        output_dictionary['s13'] = s_array[:,:,:,2].squeeze()
        output_dictionary['s21'] = s_array[:,:,:,3].squeeze()
        output_dictionary['s22'] = s_array[:,:,:,4].squeeze()
        output_dictionary['s23'] = s_array[:,:,:,5].squeeze()
        output_dictionary['s31'] = s_array[:,:,:,6].squeeze()
        output_dictionary['s32'] = s_array[:,:,:,7].squeeze()
        output_dictionary['s33'] = s_array[:,:,:,8].squeeze()
    
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


def moller(in_state, momentum, theta, phi=None, projection_state=None, 
           filename=None, diff_probability=False, state_concurrence=False, 
           stokes=False, deg_polarization=False):
    """
    Calculates differential probability, output state concurrence, Stokes 
    parameters, and degree of polarization for a tree level Moller 
    scattering process in the centre-of-mass frame. 

    Parameters
    ----------
    in_state : QuantumState
        Quantum state of incoming electrons.
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
    projection_state : QuantumState 
        Optional input of the state onto which the outgoing (scattered) 
        electron-electron state is projected, such that differential proobability
        values are adjusted accordingly. `False` by default.
    filename : str 
        If `filename` is `str` the output data dictionary is saved as a 
        Pickle file, with the string as a title. `None` by default.
    diff_probability : bool 
        If `diff_probability` is `True` an array of values of the spin-averaged 
        sum of |M|^2 (the squared magnitude of amplitudes) is output. The array 
        runs over all input `momentum`, `theta`, and `phi` values. `False` by 
        default.
    state_concurrence : bool 
        If `state_concurrence` is `True` an array of concurrence values of the 
        scattered state is output. The array runs over all input `momentum`, 
        `theta`, and `phi` values. `False` by default.
    stokes : bool 
        If `state_concurrence` is `True` an array of the Stokes parameters of 
        the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    deg_polarization : bool 
        If `state_concurrence` is `True` an array of the degree of polarization 
        values of the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    n_units : int
        Redefines all global variables, such that all calculations are done in 
        XeV units, where the placeholder X stands for the unit prefix 
        corresponding to an order of magnitude of n according to the standard 
        convention.

    Returns
    -------
    output_dictionary : dict
        Dictionary of `diff_probability`, `state_concurrence`, `stokes`, 
        `deg_polarization` arrays.
    ---------------------------------------------------------------------------
    
    """

    # Raise a type error if in_state and/or projection_state are not of 
    # `QuantumState` type:
    if not isinstance(in_state, QuantumState):
        raise TypeError("Expected 'in_state' to be of type `QuantumState`," \
                        + f" but got {type(in_state).__name__}.")
    
    if projection_state is not None:
        if not isinstance(projection_state, QuantumState):
            raise TypeError("Expected 'projection_state' to be of type"\
            + f" `QuantumState`, but got {type(projection_state).__name__}.")
        
        if np.shape(projection_state.rho) != (4, 4):
            raise TypeError("Expected 'projection_state' to be"\
            + f" of shape (4,4), but got {np.shape(projection_state.rho)}.")
    
    if np.shape(in_state.rho) != (4, 4):
        raise TypeError("Expected 'in_state' to be of"\
                        + f" shape (4,4), but got {np.shape(in_state.rho)}.")


    # Raise a type error if `momentum`, `theta`, and `phi` are not of lists
    # or NumPy arrays:
    if momentum is not None:
        if not isinstance(momentum, (list, np.ndarray)):
            raise TypeError("Expected 'energy' to be a list or a NumPy array,"\
                            + f" but got {type(momentum).__name__}.")
        
    if theta is not None:
        if not isinstance(theta, (list, np.ndarray)):
            raise TypeError("Expected 'theta' to be a list or a NumPy array,"\
                            + f" but got {type(theta).__name__}.")
    
    if phi is not None:
        if not isinstance(phi, (list, np.ndarray)):
            raise TypeError(f"Expected 'phi' to be a list or a NumPy array"\
                            + f", but got {type(phi).__name__}.")
    elif phi == None:
        phi = [0]

    # Convert input array_like objects to NumPy arrays:
    momentum = np.array(momentum)
    theta = np.array(theta)
    phi = np.array(phi)

    # Array of electron's and muon's energy values 
    energy_e = np.sqrt(momentum**2 + m_e**2)
    
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
    
    # Initialise output values array if given not False:
    # Update the "number of True inputs" parameter:
    if diff_probability != False:

        num_output += 1

        dW_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if state_concurrence != False:

        num_output += 1

        conc_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if stokes != False:

        num_output += 9

        s_array = np.array([[[[0 + 0j for _ in range(9)] \
            for _ in range(len(phi))] for _ in range(len(theta))] \
            for _ in range(len(momentum))])

    if deg_polarization != False:

        num_output += 1

        pol_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    # Define a list of 12 empty lists:
    output_data = empty_lists(num_output)

    # Loop over the size of the list of momenta:
    for momentum_index in range(len(momentum)):

        # Loop over the size of the list of theta angles:
        for theta_index in range(len(theta)):

            # Loop over the size of the list of phi angles:
            for phi_index in range(len(phi)):

                p_electron1_in = FourVector(energy_e[momentum_index], \
                    momentum[momentum_index], 0, 0)
                
                p_electron2_in = FourVector(energy_e[momentum_index], \
                    momentum[momentum_index], np.pi, 0)
                
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
                    
                    # Loop over all helicity configurations in a single array :
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
                        u_electron1_in = electron1_in.polarization.bispinor
        
                        u_electron1_out = electron1_out.polarization.bispinor

                        u_electron2_in = electron2_in.polarization.bispinor
        
                        u_electron2_out = electron2_out.polarization.bispinor
                        
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
                out_state = QuantumState.out_state(in_state, M_matrix)
                
                # Initialise empty lists for output ictionary and .pkl file:
                quantities = []
                keys = []

                # Calculate and append all output quantities if not False:
                if diff_probability != False:

                    dW_array[momentum_index,theta_index,phi_index] = \
                        differential_probability(out_state, projection_state)

                if state_concurrence != False:

                    conc_array[momentum_index,theta_index,phi_index] \
                        = concurrence(out_state)

                if deg_polarization != False:
                    
                    pol_array[momentum_index,theta_index,phi_index] = \
                        deg_of_polarization(out_state)

                if stokes != False:

                    s_array[momentum_index,theta_index,phi_index,0] \
                          = stokes_parameter(out_state, [1, 1])
                    s_array[momentum_index,theta_index,phi_index,1] \
                          = stokes_parameter(out_state, [1, 2])
                    s_array[momentum_index,theta_index,phi_index,2] \
                          = stokes_parameter(out_state, [1, 3])
                    s_array[momentum_index,theta_index,phi_index,3] \
                          = stokes_parameter(out_state, [2, 1])
                    s_array[momentum_index,theta_index,phi_index,4] \
                          = stokes_parameter(out_state, [2, 2])
                    s_array[momentum_index,theta_index,phi_index,5] \
                          = stokes_parameter(out_state, [2, 3])
                    s_array[momentum_index,theta_index,phi_index,6] \
                          = stokes_parameter(out_state, [3, 1])
                    s_array[momentum_index,theta_index,phi_index,7] \
                          = stokes_parameter(out_state, [3, 2])
                    s_array[momentum_index,theta_index,phi_index,8] \
                          = stokes_parameter(out_state, [3, 3])
    
    # Append all output quantities:
    if diff_probability != False:

        quantities.append(dW_array)
        keys.append('diff_probability')
        output_dictionary['diff_probability'] = dW_array.squeeze()

    if state_concurrence != False:

        quantities.append(conc_array)
        keys.append('state_concurrence')
        output_dictionary['state_concurrence'] = conc_array.squeeze()

    if deg_polarization != False:

        quantities.append(pol_array)
        keys.append('deg_polarization')
        output_dictionary['deg_polarization'] = pol_array.squeeze()

    if stokes != False:

        quantities.extend([s_array[:,:,:,0], s_array[:,:,:,1], \
            s_array[:,:,:,2], s_array[:,:,:,3], s_array[:,:,:,4], \
            s_array[:,:,:,5], s_array[:,:,:,6], s_array[:,:,:,7], \
            s_array[:,:,:,8]])
                    
        keys.extend(['s11', 's12', 's13', 's21', 's22', 's23', \
                                 's31', 's32', 's33'])
        
        output_dictionary['s11'] = s_array[:,:,:,0].squeeze()
        output_dictionary['s12'] = s_array[:,:,:,1].squeeze()
        output_dictionary['s13'] = s_array[:,:,:,2].squeeze()
        output_dictionary['s21'] = s_array[:,:,:,3].squeeze()
        output_dictionary['s22'] = s_array[:,:,:,4].squeeze()
        output_dictionary['s23'] = s_array[:,:,:,5].squeeze()
        output_dictionary['s31'] = s_array[:,:,:,6].squeeze()
        output_dictionary['s32'] = s_array[:,:,:,7].squeeze()
        output_dictionary['s33'] = s_array[:,:,:,8].squeeze()
    
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


def electron_positron_annihilation(in_state, momentum, theta, phi=None, 
                                   projection_state=None, filename=None,
                                   diff_probability=False, 
                                   state_concurrence=False, stokes=False, 
                                   deg_polarization=False):
    """
    Calculates differential probability, output state concurrence, Stokes 
    parameters, and degree of polarization for a tree level electron-positron
    annihilation process in the centre-of-mass frame. 

    Parameters
    ----------
    in_state : QuantumState
        Quantum state of incoming electron and positron particles.
    momentum : array_like 
        Values of the incoming particles' three-momentum magnitude (in the 
        centre-of-mass frame) over which all output quantities are calculated.
    theta : array_like 
        Values of the polar angle (angle between the outgoing and incoming 
        photon three-momenta) over which all output quantities are 
        calculated.
    phi : array_like 
        Values of the azimuthal angle (angle between the x-axis of the 
        scattering coordinate system and the projection of the outgoing 
        photon's three-momentum onto the x-y plane) over which all output 
        quantities are calculated.
    projection_state : QuantumState 
        Optional input of the state onto which the outgoing (scattered) 
        photon-photon state is projected, such that differential proobability
        values are adjusted accordingly. `False` by default.
    filename : str 
        If `filename` is `str` the output data dictionary is saved as a 
        Pickle file, with the string as a title. `None` by default.
    diff_probability : bool 
        If `diff_probability` is `True` an array of values of the spin-averaged 
        sum of |M|^2 (the squared magnitude of amplitudes) is output. The array 
        runs over all input `momentum`, `theta`, and `phi` values. `False` by 
        default.
    state_concurrence : bool 
        If `state_concurrence` is `True` an array of concurrence values of the 
        scattered state is output. The array runs over all input `momentum`, 
        `theta`, and `phi` values. `False` by default.
    stokes : bool 
        If `state_concurrence` is `True` an array of the Stokes parameters of 
        the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    deg_polarization : bool 
        If `state_concurrence` is `True` an array of the degree of polarization 
        values of the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    n_units : int
        Redefines all global variables, such that all calculations are done in 
        XeV units, where the placeholder X stands for the unit prefix 
        corresponding to an order of magnitude of n according to the standard 
        convention.

    Returns
    -------
    output_dictionary : dict
        Dictionary of `diff_probability`, `state_concurrence`, `stokes`, 
        `deg_polarization` arrays.
    ---------------------------------------------------------------------------
    
    """

    # Raise a type error if in_state and/or projection_state are not of 
    # `QuantumState` type:
    if not isinstance(in_state, QuantumState):
        raise TypeError("Expected 'in_state' to be of type `QuantumState`," \
                        + f" but got {type(in_state).__name__}.")
    
    if projection_state is not None:
        if not isinstance(projection_state, QuantumState):
            raise TypeError("Expected 'projection_state' to be of type"\
            + f" `QuantumState`, but got {type(projection_state).__name__}.")
        
        if np.shape(projection_state.rho) != (4, 4):
            raise TypeError("Expected 'projection_state' to be"\
            + f" of shape (4,4), but got {np.shape(projection_state.rho)}.")
    
    if np.shape(in_state.rho) != (4, 4):
        raise TypeError("Expected 'in_state' to be of"\
                        + f" shape (4,4), but got {np.shape(in_state.rho)}.")


    # Raise a type error if `momentum`, `theta`, and `phi` are not of lists
    # or NumPy arrays:
    if momentum is not None:
        if not isinstance(momentum, (list, np.ndarray)):
            raise TypeError("Expected 'energy' to be a list or a NumPy array,"\
                        + f" but got {type(momentum).__name__}.")
        
    if theta is not None:
        if not isinstance(theta, (list, np.ndarray)):
            raise TypeError("Expected 'theta' to be a list or a NumPy array,"\
                        + f" but got {type(theta).__name__}.")
    
    if phi is not None:
        if not isinstance(phi, (list, np.ndarray)):
            raise TypeError(f"Expected 'phi' to be a list or a NumPy array"\
                        + f", but got {type(phi).__name__}.")
    elif phi == None:
        phi = [0]

    # Convert input array_like objects to NumPy arrays:
    momentum = np.array(momentum)
    theta = np.array(theta)
    phi = np.array(phi)

    # Array of the incoming electron's energy values 
    energy = np.sqrt(momentum**2 + m_e**2)
    
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
    
    # Initialise output values array if given not False:
    # Update the "number of True inputs" parameter:
    if diff_probability != False:

        num_output += 1

        dW_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if state_concurrence != False:

        num_output += 1

        conc_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if stokes != False:

        num_output += 9

        s_array = np.array([[[[0 + 0j for _ in range(9)] \
            for _ in range(len(phi))] for _ in range(len(theta))] \
            for _ in range(len(momentum))])

    if deg_polarization != False:

        num_output += 1

        pol_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    # Define a list of 12 empty lists:
    output_data = empty_lists(num_output)

    # Loop over the size of the list of momenta:
    for momentum_index in range(len(momentum)):

        # Loop over the size of the list of theta angles:
        for theta_index in range(len(theta)):

            # Loop over the size of the list of phi angles:
            for phi_index in range(len(phi)):

                p_electron_in = FourVector(energy[momentum_index], \
                    momentum[momentum_index], 0, 0)
                
                p_positron_in = FourVector(energy[momentum_index], \
                    momentum[momentum_index], np.pi, 0)
                
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
                    
                    # Loop over all helicity configurations in a single array :
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

                # Find the scattered electron-photon state:
                out_state = QuantumState.out_state(in_state, M_matrix)
                
                # Initialise empty lists for output ictionary and .pkl file:
                quantities = []
                keys = []

                # Calculate and append all output quantities if not False:
                if diff_probability != False:

                    dW_array[momentum_index,theta_index,phi_index] = \
                        differential_probability(out_state, projection_state)

                if state_concurrence != False:

                    conc_array[momentum_index,theta_index,phi_index] \
                        = concurrence(out_state)

                if deg_polarization != False:
                    
                    pol_array[momentum_index,theta_index,phi_index] = \
                        deg_of_polarization(out_state)

                if stokes != False:

                    s_array[momentum_index,theta_index,phi_index,0] \
                          = stokes_parameter(out_state, [1, 1])
                    s_array[momentum_index,theta_index,phi_index,1] \
                          = stokes_parameter(out_state, [1, 2])
                    s_array[momentum_index,theta_index,phi_index,2] \
                          = stokes_parameter(out_state, [1, 3])
                    s_array[momentum_index,theta_index,phi_index,3] \
                          = stokes_parameter(out_state, [2, 1])
                    s_array[momentum_index,theta_index,phi_index,4] \
                          = stokes_parameter(out_state, [2, 2])
                    s_array[momentum_index,theta_index,phi_index,5] \
                          = stokes_parameter(out_state, [2, 3])
                    s_array[momentum_index,theta_index,phi_index,6] \
                          = stokes_parameter(out_state, [3, 1])
                    s_array[momentum_index,theta_index,phi_index,7] \
                          = stokes_parameter(out_state, [3, 2])
                    s_array[momentum_index,theta_index,phi_index,8] \
                          = stokes_parameter(out_state, [3, 3])
    
    # Append all output quantities:
    if diff_probability != False:

        quantities.append(dW_array)
        keys.append('diff_probability')
        output_dictionary['diff_probability'] = dW_array.squeeze()

    if state_concurrence != False:

        quantities.append(conc_array)
        keys.append('state_concurrence')
        output_dictionary['state_concurrence'] = conc_array.squeeze()

    if deg_polarization != False:

        quantities.append(pol_array)
        keys.append('deg_polarization')
        output_dictionary['deg_polarization'] = pol_array.squeeze()

    if stokes != False:

        quantities.extend([s_array[:,:,:,0], s_array[:,:,:,1], \
            s_array[:,:,:,2], s_array[:,:,:,3], s_array[:,:,:,4], \
            s_array[:,:,:,5], s_array[:,:,:,6], s_array[:,:,:,7], \
            s_array[:,:,:,8]])
                    
        keys.extend(['s11', 's12', 's13', 's21', 's22', 's23', \
                                 's31', 's32', 's33'])
        
        output_dictionary['s11'] = s_array[:,:,:,0].squeeze()
        output_dictionary['s12'] = s_array[:,:,:,1].squeeze()
        output_dictionary['s13'] = s_array[:,:,:,2].squeeze()
        output_dictionary['s21'] = s_array[:,:,:,3].squeeze()
        output_dictionary['s22'] = s_array[:,:,:,4].squeeze()
        output_dictionary['s23'] = s_array[:,:,:,5].squeeze()
        output_dictionary['s31'] = s_array[:,:,:,6].squeeze()
        output_dictionary['s32'] = s_array[:,:,:,7].squeeze()
        output_dictionary['s33'] = s_array[:,:,:,8].squeeze()
    
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


def electron_muon(in_state, momentum, theta, phi=None, projection_state=None, 
                  filename=None, diff_probability=False, 
                  state_concurrence=False, stokes=False, 
                  deg_polarization=False):
    """
    Calculates differential probability, output state concurrence, Stokes 
    parameters, and degree of polarization for a tree level electron-muon 
    scattering process in the centre-of-mass frame. 

    Parameters
    ----------
    in_state : QuantumState
        Quantum state of incoming electron and muon particles.
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
    projection_state : QuantumState 
        Optional input of the state onto which the outgoing (scattered) 
        electron-muon state is projected, such that differential proobability
        values are adjusted accordingly. `False` by default.
    filename : str 
        If `filename` is `str` the output data dictionary is saved as a 
        Pickle file, with the string as a title. `None` by default.
    diff_probability : bool 
        If `diff_probability` is `True` an array of values of the spin-averaged 
        sum of |M|^2 (the squared magnitude of amplitudes) is output. The array 
        runs over all input `momentum`, `theta`, and `phi` values. `False` by 
        default.
    state_concurrence : bool 
        If `state_concurrence` is `True` an array of concurrence values of the 
        scattered state is output. The array runs over all input `momentum`, 
        `theta`, and `phi` values. `False` by default.
    stokes : bool 
        If `state_concurrence` is `True` an array of the Stokes parameters of 
        the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    deg_polarization : bool 
        If `state_concurrence` is `True` an array of the degree of polarization 
        values of the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    n_units : int
        Redefines all global variables, such that all calculations are done in 
        XeV units, where the placeholder X stands for the unit prefix 
        corresponding to an order of magnitude of n according to the standard 
        convention.

    Returns
    -------
    output_dictionary : dict
        Dictionary of `diff_probability`, `state_concurrence`, `stokes`, 
        `deg_polarization` arrays.
    ---------------------------------------------------------------------------
    
    """

    # Raise a type error if in_state and/or projection_state are not of 
    # `QuantumState` type:
    if not isinstance(in_state, QuantumState):
        raise TypeError("Expected 'in_state' to be of type `QuantumState`," \
                        + f" but got {type(in_state).__name__}.")
    
    if projection_state is not None:
        if not isinstance(projection_state, QuantumState):
            raise TypeError("Expected 'projection_state' to be of type"\
            + f" `QuantumState`, but got {type(projection_state).__name__}.")
        
        if np.shape(projection_state.rho) != (4, 4):
            raise TypeError("Expected 'projection_state' to be"\
            + f" of shape (4,4), but got {np.shape(projection_state.rho)}.")
    
    if np.shape(in_state.rho) != (4, 4):
        raise TypeError("Expected 'in_state' to be of"\
                        + f" shape (4,4), but got {np.shape(in_state.rho)}.")


    # Raise a type error if `momentum`, `theta`, and `phi` are not of lists
    # or NumPy arrays:
    if momentum is not None:
        if not isinstance(momentum, (list, np.ndarray)):
            raise TypeError("Expected 'energy' to be a list or a NumPy array,"\
                        + f" but got {type(momentum).__name__}.")
        
    if theta is not None:
        if not isinstance(theta, (list, np.ndarray)):
            raise TypeError("Expected 'theta' to be a list or a NumPy array,"\
                        + f" but got {type(theta).__name__}.")
    
    if phi is not None:
        if not isinstance(phi, (list, np.ndarray)):
            raise TypeError(f"Expected 'phi' to be a list or a NumPy array"\
                        + f", but got {type(phi).__name__}.")
    elif phi == None:
        phi = [0]

    # Convert input array_like objects to NumPy arrays:
    momentum = np.array(momentum)
    theta = np.array(theta)
    phi = np.array(phi)

    # Array of electron's and muon's energy values 
    energy_e = np.sqrt(momentum**2 + m_e**2)
    energy_mu = np.sqrt(momentum**2 + m_mu**2)
    
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
    
    # Initialise output values array if given not False:
    # Update the "number of True inputs" parameter:
    if diff_probability != False:

        num_output += 1

        dW_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if state_concurrence != False:

        num_output += 1

        conc_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if stokes != False:

        num_output += 9

        s_array = np.array([[[[0 + 0j for _ in range(9)] \
            for _ in range(len(phi))] for _ in range(len(theta))] \
            for _ in range(len(momentum))])

    if deg_polarization != False:

        num_output += 1

        pol_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    # Define a list of 12 empty lists:
    output_data = empty_lists(num_output)

    # Loop over the size of the list of momenta:
    for momentum_index in range(len(momentum)):

        # Loop over the size of the list of theta angles:
        for theta_index in range(len(theta)):

            # Loop over the size of the list of phi angles:
            for phi_index in range(len(phi)):

                p_electron_in = FourVector(energy_e[momentum_index], \
                    momentum[momentum_index], 0, 0)
                
                p_muon_in = FourVector(energy_mu[momentum_index], \
                    momentum[momentum_index], np.pi, 0)
                
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
                    
                    # Loop over all helicity configurations in a single array :
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
                        M_matrix_term = -1j * lorentzian_product(J_e, J_mu) \
                            / lorentzian_product(q_virtual_t, q_virtual_t)
                        
                        # Append to M_matrix_row
                        M_matrix_row.append(M_matrix_term)

                    # Append to the array of all amplitudes:
                    M_matrix.append(M_matrix_row)

                # Find the scattered electron-positron state:
                out_state = QuantumState.out_state(in_state, M_matrix)
                
                # Initialise empty lists for output ictionary and .pkl file:
                quantities = []
                keys = []

                # Calculate and append all output quantities if not False:
                if diff_probability != False:

                    dW_array[momentum_index,theta_index,phi_index] = \
                        differential_probability(out_state, projection_state)

                if state_concurrence != False:

                    conc_array[momentum_index,theta_index,phi_index] \
                        = concurrence(out_state)

                if deg_polarization != False:
                    
                    pol_array[momentum_index,theta_index,phi_index] = \
                        deg_of_polarization(out_state)

                if stokes != False:

                    s_array[momentum_index,theta_index,phi_index,0] \
                          = stokes_parameter(out_state, [1, 1])
                    s_array[momentum_index,theta_index,phi_index,1] \
                          = stokes_parameter(out_state, [1, 2])
                    s_array[momentum_index,theta_index,phi_index,2] \
                          = stokes_parameter(out_state, [1, 3])
                    s_array[momentum_index,theta_index,phi_index,3] \
                          = stokes_parameter(out_state, [2, 1])
                    s_array[momentum_index,theta_index,phi_index,4] \
                          = stokes_parameter(out_state, [2, 2])
                    s_array[momentum_index,theta_index,phi_index,5] \
                          = stokes_parameter(out_state, [2, 3])
                    s_array[momentum_index,theta_index,phi_index,6] \
                          = stokes_parameter(out_state, [3, 1])
                    s_array[momentum_index,theta_index,phi_index,7] \
                          = stokes_parameter(out_state, [3, 2])
                    s_array[momentum_index,theta_index,phi_index,8] \
                          = stokes_parameter(out_state, [3, 3])
    
    # Append all output quantities:
    if diff_probability != False:

        quantities.append(dW_array)
        keys.append('diff_probability')
        output_dictionary['diff_probability'] = dW_array.squeeze()

    if state_concurrence != False:

        quantities.append(conc_array)
        keys.append('state_concurrence')
        output_dictionary['state_concurrence'] = conc_array.squeeze()

    if deg_polarization != False:

        quantities.append(pol_array)
        keys.append('deg_polarization')
        output_dictionary['deg_polarization'] = pol_array.squeeze()

    if stokes != False:

        quantities.extend([s_array[:,:,:,0], s_array[:,:,:,1], \
            s_array[:,:,:,2], s_array[:,:,:,3], s_array[:,:,:,4], \
            s_array[:,:,:,5], s_array[:,:,:,6], s_array[:,:,:,7], \
            s_array[:,:,:,8]])
                    
        keys.extend(['s11', 's12', 's13', 's21', 's22', 's23', \
                                 's31', 's32', 's33'])
        
        output_dictionary['s11'] = s_array[:,:,:,0].squeeze()
        output_dictionary['s12'] = s_array[:,:,:,1].squeeze()
        output_dictionary['s13'] = s_array[:,:,:,2].squeeze()
        output_dictionary['s21'] = s_array[:,:,:,3].squeeze()
        output_dictionary['s22'] = s_array[:,:,:,4].squeeze()
        output_dictionary['s23'] = s_array[:,:,:,5].squeeze()
        output_dictionary['s31'] = s_array[:,:,:,6].squeeze()
        output_dictionary['s32'] = s_array[:,:,:,7].squeeze()
        output_dictionary['s33'] = s_array[:,:,:,8].squeeze()
    
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


def electron_positron_to_muon_antimuon(in_state, momentum, theta, phi=None,
                                       projection_state=None, filename=None,
                                       diff_probability=False, 
                                       state_concurrence=False, stokes=False, 
                                       deg_polarization=False):
    """
    Calculates differential probability, output state concurrence, Stokes 
    parameters, and degree of polarization for a tree level electron-positron
    to muon-antmuon creation-annihilation process in the centre-of-mass frame. 

    Parameters
    ----------
    in_state : QuantumState
        Quantum state of incoming electron and positron particles.
    momentum : array_like 
        Values of the incoming particles' three-momentum magnitude (in the 
        centre-of-mass frame) over which all output quantities are calculated.
    theta : array_like 
        Values of the polar angle (angle between the incoming electron and outgoing muon
        three-momenta) over which all output quantities are calculated.
    phi : array_like 
        Values of the azimuthal angle (angle between the x-axis of the 
        scattering coordinate system and the projection of the outgoing 
        muon's three-momentum onto the x-y plane) over which all output 
        quantities are calculated.
    projection_state : QuantumState 
        Optional input of the state onto which the outgoing (scattered) 
        muon-antimuon state is projected, such that differential proobability
        values are adjusted accordingly. `False` by default.
    filename : str 
        If `filename` is `str` the output data dictionary is saved as a 
        Pickle file, with the string as a title. `None` by default.
    diff_probability : bool 
        If `diff_probability` is `True` an array of values of the spin-averaged 
        sum of |M|^2 (the squared magnitude of amplitudes) is output. The array 
        runs over all input `momentum`, `theta`, and `phi` values. `False` by 
        default.
    state_concurrence : bool 
        If `state_concurrence` is `True` an array of concurrence values of the 
        scattered state is output. The array runs over all input `momentum`, 
        `theta`, and `phi` values. `False` by default.
    stokes : bool 
        If `state_concurrence` is `True` an array of the Stokes parameters of 
        the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    deg_polarization : bool 
        If `state_concurrence` is `True` an array of the degree of polarization 
        values of the scattered state is output. The array runs over all input 
        `momentum`, `theta`, and `phi` values. `False` by default.
    n_units : int
        Redefines all global variables, such that all calculations are done in 
        XeV units, where the placeholder X stands for the unit prefix 
        corresponding to an order of magnitude of n according to the standard 
        convention.

    Returns
    -------
    output_dictionary : dict
        Dictionary of `diff_probability`, `state_concurrence`, `stokes`, 
        `deg_polarization` arrays.
    ---------------------------------------------------------------------------
    
    """

    # Raise a type error if in_state and/or projection_state are not of 
    # `QuantumState` type:
    if not isinstance(in_state, QuantumState):
        raise TypeError("Expected 'in_state' to be of type `QuantumState`," \
                        + f" but got {type(in_state).__name__}.")
    
    if projection_state is not None:
        if not isinstance(projection_state, QuantumState):
            raise TypeError("Expected 'projection_state' to be of type"\
            + f" `QuantumState`, but got {type(projection_state).__name__}.")
        
        if np.shape(projection_state.rho) != (4, 4):
            raise TypeError("Expected 'projection_state' to be"\
            + f" of shape (4,4), but got {np.shape(projection_state.rho)}.")
    
    if np.shape(in_state.rho) != (4, 4):
        raise TypeError("Expected 'in_state' to be of"\
                        + f" shape (4,4), but got {np.shape(in_state.rho)}.")


    # Raise a type error if `momentum`, `theta`, and `phi` are not of lists
    # or NumPy arrays:
    if momentum is not None:
        if not isinstance(momentum, (list, np.ndarray)):
            raise TypeError("Expected 'energy' to be a list or a NumPy array,"\
                        + f" but got {type(momentum).__name__}.")
        
    if theta is not None:
        if not isinstance(theta, (list, np.ndarray)):
            raise TypeError("Expected 'theta' to be a list or a NumPy array,"\
                        + f" but got {type(theta).__name__}.")
    
    if phi is not None:
        if not isinstance(phi, (list, np.ndarray)):
            raise TypeError(f"Expected 'phi' to be a list or a NumPy array"\
                        + f", but got {type(phi).__name__}.")
    elif phi == None:
        phi = [0]
        
    if momentum.min() < np.sqrt(MUON_MASS**2-ELECTRON_MASS**2):
        raise ValueError("Input momentum values" \
    " must be larger than np.sqrt(MUON_MASS**2-ELECTRON_MASS**2).")

    # Convert input array_like objects to NumPy arrays:
    momentum = np.array(momentum)
    theta = np.array(theta)
    phi = np.array(phi)

    # Array of electron's and muon's energy values 
    energy_e = np.sqrt(momentum**2 + m_e**2)
    
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
    
    # Initialise output values array if given not False:
    # Update the "number of True inputs" parameter:
    if diff_probability != False:

        num_output += 1

        dW_array = np.array([[[1e-20 + 1e-20j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if state_concurrence != False:

        num_output += 1

        conc_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    if stokes != False:

        num_output += 9

        s_array = np.array([[[[0 + 0j for _ in range(9)] \
            for _ in range(len(phi))] for _ in range(len(theta))] \
            for _ in range(len(momentum))])

    if deg_polarization != False:

        num_output += 1

        pol_array = np.array([[[0 + 0j for _ in range(len(phi))] \
            for _ in range(len(theta))] for _ in range(len(momentum))])

    # Define a list of 12 empty lists:
    output_data = empty_lists(num_output)

    # Loop over the size of the list of momenta:
    for momentum_index in range(len(momentum)):

        # Loop over the size of the list of theta angles:
        for theta_index in range(len(theta)):

            # Loop over the size of the list of phi angles:
            for phi_index in range(len(phi)):

                p_electron_in = FourVector(energy_e[momentum_index], \
                    momentum[momentum_index], 0, 0)
                
                p_positron_in = FourVector(energy_e[momentum_index], \
                    momentum[momentum_index], np.pi, 0)
                
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
                    
                    # Loop over all helicity configurations in a single array :
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
                        M_matrix_term = -1j * lorentzian_product(J_e, J_mu) \
                            / lorentzian_product(q_virtual_s, q_virtual_s)
                        
                        # Append to M_matrix_row
                        M_matrix_row.append(M_matrix_term)

                    # Append to the array of all amplitudes:
                    M_matrix.append(M_matrix_row)

                # Find the scattered electron-positron state:
                out_state = QuantumState.out_state(in_state, M_matrix)
                
                # Initialise empty lists for output ictionary and .pkl file:
                quantities = []
                keys = []

                # Calculate and append all output quantities if not False:
                if diff_probability != False:

                    dW_array[momentum_index,theta_index,phi_index] = \
                        differential_probability(out_state, projection_state)

                if state_concurrence != False:

                    conc_array[momentum_index,theta_index,phi_index] \
                        = concurrence(out_state)

                if deg_polarization != False:
                    
                    pol_array[momentum_index,theta_index,phi_index] = \
                        deg_of_polarization(out_state)

                if stokes != False:

                    s_array[momentum_index,theta_index,phi_index,0] \
                          = stokes_parameter(out_state, [1, 1])
                    s_array[momentum_index,theta_index,phi_index,1] \
                          = stokes_parameter(out_state, [1, 2])
                    s_array[momentum_index,theta_index,phi_index,2] \
                          = stokes_parameter(out_state, [1, 3])
                    s_array[momentum_index,theta_index,phi_index,3] \
                          = stokes_parameter(out_state, [2, 1])
                    s_array[momentum_index,theta_index,phi_index,4] \
                          = stokes_parameter(out_state, [2, 2])
                    s_array[momentum_index,theta_index,phi_index,5] \
                          = stokes_parameter(out_state, [2, 3])
                    s_array[momentum_index,theta_index,phi_index,6] \
                          = stokes_parameter(out_state, [3, 1])
                    s_array[momentum_index,theta_index,phi_index,7] \
                          = stokes_parameter(out_state, [3, 2])
                    s_array[momentum_index,theta_index,phi_index,8] \
                          = stokes_parameter(out_state, [3, 3])
    
    # Append all output quantities:
    if diff_probability != False:

        quantities.append(dW_array)
        keys.append('diff_probability')
        output_dictionary['diff_probability'] = dW_array.squeeze()

    if state_concurrence != False:

        quantities.append(conc_array)
        keys.append('state_concurrence')
        output_dictionary['state_concurrence'] = conc_array.squeeze()

    if deg_polarization != False:

        quantities.append(pol_array)
        keys.append('deg_polarization')
        output_dictionary['deg_polarization'] = pol_array.squeeze()

    if stokes != False:

        quantities.extend([s_array[:,:,:,0], s_array[:,:,:,1], \
            s_array[:,:,:,2], s_array[:,:,:,3], s_array[:,:,:,4], \
            s_array[:,:,:,5], s_array[:,:,:,6], s_array[:,:,:,7], \
            s_array[:,:,:,8]])
                    
        keys.extend(['s11', 's12', 's13', 's21', 's22', 's23', \
                                 's31', 's32', 's33'])
        
        output_dictionary['s11'] = s_array[:,:,:,0].squeeze()
        output_dictionary['s12'] = s_array[:,:,:,1].squeeze()
        output_dictionary['s13'] = s_array[:,:,:,2].squeeze()
        output_dictionary['s21'] = s_array[:,:,:,3].squeeze()
        output_dictionary['s22'] = s_array[:,:,:,4].squeeze()
        output_dictionary['s23'] = s_array[:,:,:,5].squeeze()
        output_dictionary['s31'] = s_array[:,:,:,6].squeeze()
        output_dictionary['s32'] = s_array[:,:,:,7].squeeze()
        output_dictionary['s33'] = s_array[:,:,:,8].squeeze()
    
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

