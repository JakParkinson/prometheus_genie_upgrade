import numpy as np
def inject_particles_in_cylinder(primary_set, ## neutrino information
                        prometheus_set, ## child particle information
                        cylinder_radius=500, ## meters
                        cylinder_height = 1000, # meters
                        cylinder_center = (0, 0, 0), # meters
                        detector_offset = (10.12, -8.56, -2005.37)## Specific to Upgrade! need to use printed offset if using other detector
                        ):
    """
    Sample a random point in a cylinder. Then assign the interaction vertex to sampled position and offset the 
    position of child particles by this sampled point.
    IMPORTANT: ASSUMES INTERACTION VERTEX IS AT (0,0,0), which is default by gevgen... 
    """
    if len(primary_set) != len(prometheus_set):
        raise ValueError('Length of primary set (neutrino informaiton) does not equal length of prometheus set (child particles) !!') 

    n_events = len(primary_set)
    ## cyyyylinder:
    r = np.sqrt(np.random.uniform(0, 1, n_events)) * cylinder_radius
    theta = np.random.uniform(0, 2*np.pi, n_events) ## theta in circle
    z = np.random.uniform(-cylinder_height/2, cylinder_height/2, n_events)
    cylinder_center = ( #
        cylinder_center[0] - detector_offset[0],
        cylinder_center[1] - detector_offset[1], 
        cylinder_center[2] - detector_offset[2]
    )
    ## to cartesian:
    x = r * np.cos(theta) + cylinder_center[0]
    y = r * np.sin(theta) + cylinder_center[1]
    z = z + cylinder_center[2]

    new_vertices = []
    for i in range(n_events):
        new_vertices.append((x[i], y[i], z[i], 0.0))
    primary_set['position'] = new_vertices ## this is assuming initial vertex is 0,0,0 
    primary_set['pos_x'] = [pos[0] for pos in primary_set['position']] ## not sure why we need pos x,y,z if we have position but whatever
    primary_set['pos_y'] = [pos[1] for pos in primary_set['position']]
    primary_set['pos_z'] = [pos[2] for pos in primary_set['position']]

    for i in range(n_events):
        ## kinda silly but it works, also intentionally chose to do offset rather then declaration of new position
        prometheus_set.loc[i, 'position'] += np.array([np.array(primary_set['position'].iloc[i])[0:3]] * prometheus_set.loc[i, 'position'].shape[0]) ## [0:3] excludes time    
        prometheus_set.loc[i, 'pos_x'] += np.array([np.array(primary_set['pos_x'].iloc[i])] * prometheus_set.loc[i, 'pos_x'].shape[0])
        prometheus_set.loc[i, 'pos_y'] += np.array([np.array(primary_set['pos_y'].iloc[i])] * prometheus_set.loc[i, 'pos_y'].shape[0])
        prometheus_set.loc[i, 'pos_z'] += np.array([np.array(primary_set['pos_z'].iloc[i])] * prometheus_set.loc[i, 'pos_z'].shape[0])

    return primary_set, prometheus_set
