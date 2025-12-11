import numpy as np

def rotate_particles_final(primary_set, prometheus_set):
    """
    Samples a rotation isotopically, then rotates neutrino and child particles
    using Rodrigues' rotation formula to create an isotropic flux
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    IMPORTANT: ASSUMES NEUTRINO UNIT VECTOR DIRECTION IS (0, 0, 1) - which is default in gevgen
    """
    if len(primary_set) != len(prometheus_set):
        raise ValueError('Length of primary sets do not match!')
    
    n_events = len(primary_set)
    ## think i should add an error if inital neutrino isnt [0,0,1]
    initial_neutrino = np.array([0, 0, 1]) #### assumes neutrino initial direction is 0,0,1
    
    for i in range(n_events):
        # samples the target direction isotropically
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        phi = np.random.uniform(0, 2*np.pi)
        
        # calculate target neutrino direction
        target_neutrino = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # calculate rotation matrix
        rotation_matrix = rotation_matrix_from_vectors(initial_neutrino, target_neutrino)
        
        # update primary neutrino direction
        primary_set.at[i, 'theta'] = theta
        primary_set.at[i, 'phi'] = phi
        

        theta_array = prometheus_set.loc[i, 'theta'].copy()
        phi_array = prometheus_set.loc[i, 'phi'].copy()
        
        # iterate over all the child particles
        for j in range(len(theta_array)):
            if np.isnan(theta_array[j]) or np.isnan(phi_array[j]):
                continue
                
            theta_orig = theta_array[j]
            phi_orig = phi_array[j]
            
            direction = np.array([
                np.sin(theta_array[j]) * np.cos(phi_array[j]),
                np.sin(theta_array[j]) * np.sin(phi_array[j]),
                np.cos(theta_array[j])
            ])
            
            # rotation
            rotated = np.dot(rotation_matrix, direction)
            
            theta_array[j] = np.arccos(np.clip(rotated[2], -1.0, 1.0))
            phi_array[j] = np.arctan2(rotated[1], rotated[0])
            
        # update child particle directions
        prometheus_set.at[i, 'theta'] = theta_array
        prometheus_set.at[i, 'phi'] = phi_array
        
        # # test
        # if i == 0:
        #     print("\nAfter assignment:")
        #     print(f"  First theta original: {theta_orig:.4f}")
        #     print(f"  First theta in DataFrame: {prometheus_set.loc[0, 'theta'][0]:.4f}")
        #     print(f"  Changed in DataFrame: {prometheus_set.loc[0, 'theta'][0] != theta_orig}")
    
    return primary_set, prometheus_set

def rotation_matrix_from_vectors(vec1, vec2):
    """
    calculate rotation matrix that rotates vec1 to vec2.
    buth vectors should be unit vectors.
    """
    
    v = np.cross(vec1, vec2)
    s = np.linalg.norm(v)
    c = np.dot(vec1, vec2)
    
    if np.isclose(s, 0):
        # vectors parallel or anti-parallel
        return np.eye(3)  # same direction

    
    #Rodrigues' formula in matrix form
    v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + v_x + v_x.dot(v_x) * (1 - c) / (s * s)
    return R
