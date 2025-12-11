import pandas as pd
import numpy as np
import uproot
from schema import Schema, And, Use, Optional, SchemaError

from datetime import datetime

import time
import sys
sys.path.append('../')
from prometheus import Prometheus, config
import prometheus
#from jax.config import config as jconfig
import gc
import os

import logging


from prometheus import Prometheus, config
import prometheus
import gc

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RESOURCE_DIR = f"{'/'.join(prometheus.__path__[0].split('/')[:-1])}/resources/"

#### helper functions from genie_parser.py and genie_tester.ipynb
def angle(v1: np.array, v2: np.array) -> float:
    """ Calculates the angle between two vectors in radians

    Parameters
    ----------
    v1: np.array
        vector 1
    v2: np.array
        vector 2

    Returns
    -------
    angle: float
        The calculates angle in radians
    """
    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
    return angle

def p2azimuthAndzenith(p: np.array):
    """ converts a momentum vector to azimuth and zenith angles

    Parameters
    ----------
    p: np.array
        The 3d momentum

    Returns
    -------
    float, float:
        The azimuth and zenith angles in radians.c
    """
    azimuth = angle(p[:2], np.array([0, 1]))
    zenith = angle(p[1:], np.array([0., 1]))
    return azimuth, zenith

def genie_parser(events)->pd.DataFrame:
    """ function to fetch the relevant information from genie events (in rootracker format)

    Parameters
    ----------
    events : dict
        The genie events

    Returns
    -------
    pd.DataFrame
        Data frame containing the relevant information
    """
    dic = {}
    dic['event_description'] = events['EvtCode/fString'].array(library='np')  # String describing the event
    dic['event_id'] = events['EvtNum'].array(library="np")  # Number of the event
    dic['event_children'] = events['StdHepN'].array(library="np")  # NUmber of the generated sub-particles
    dic['event_prob'] = events['EvtProb'].array(library="np")  # Probability of the event
    dic['event_xsec'] = events['EvtXSec'].array(library="np")  # Total xsec of the event
    dic['event_pdg_id'] = events['StdHepPdg'].array(library="np")  # PDG ids of all produced particles
    dic['event_momenta'] = events['StdHepP4'].array(library="np")  # Momenta of the particles
    tmp = events['EvtVtx'].array(library="np")  # Position of the particles
    dic['event_vertex'] = [np.array(vtx) for vtx in tmp]
    dic['event_coords'] = events['StdHepX4'].array(library="np")  # Position of the particles
    dic['event_weight'] = events['EvtWght'].array(library='np')  # Weight of the events
    tmp = events['StdHepStatus'].array(library="np")  # Status of the particle
    # Converting the codes
    particle_dic = {
        0: 'initial',
        1: 'final',
        2: 'intermediate',
        3: 'decayed',
        11: 'nucleon target',
        12: 'DIS pre-frag hadronic state',
        13: 'resonance',
        14: 'hadron in nucleus',
        15: 'final nuclear',
        16: 'nucleon cluster target',
    }
    new_arr = np.array([[
        particle_dic[particle] for particle in event
    ] for event in tmp], dtype=object)
    dic['event_status'] = new_arr
    return pd.DataFrame.from_dict(dic)

def genie2prometheus(parsed_events: pd.DataFrame):
    """ reformats parsed GENIE events into a usable format for PROMETHEUS
    NOTES: Create a standardized scheme function. This could then be used as an interface to PROMETHEUS
    for any injector. E.g. a user would only need to create a function to translate their injector output to the scheme format.

    Parameters
    ----------
    parsed_events: pd.DataFrame
        Dataframe object containing all the relevant (and additional) information needed by PROMETHEUS

    Returns
    -------
    particles: pd.DataFrame
        Fromatted data set with values which can be used directly.
    injection: pd.Dataframe
        The injected particle in the same format
    """
    # TODO: Use a more elegant definition to couple to the particle class
    primaries = {}
    event_set  = {}
    for index, event in parsed_events.iterrows():
        if 'CC' in event.event_descr:
            event_type = 'CC'
        elif 'NC' in event.event_descr:
            event_type = 'NC'
        else:
            event_type = 'other'
        azimuth, zenith =  p2azimuthAndzenith(event.init_inj_p)
        primary_particle = {
            "primary": True,  # If this thing is a primary particle
            "e": event.init_inj_e,  # The energy in GeV,  # NOTE: Should this be kinetic energy?
            "pdg_code": event.init_inj_id,  # The PDG id
            "interaction": event_type,  # Should we use ints to define interactions or strings?
            "theta": zenith,  # It's zenith angle
            "phi": azimuth,  # It's azimuth angle
            "bjorken_x": -1,  # Bjorken x variable
            "bjorken_y": -1,  # Bjorken y variable
            "pos_x": event.event_vertex[0],  # position x (in detector coordinates)
            "pos_y": event.event_vertex[1],  # position y (in detector coordinates)
            "pos_z": event.event_vertex[2],  # position z (in detector coordinates)
            "position": event.event_vertex[:3],  # 3d position
            "column_depth": -1,  # Column depth where the interaction happened
            'custom_info': event.event_descr  # Additional information the user can add as a string. # TODO: This should be handed to the propagators
        }
        # TODO: Optimize
        angles = np.array([p2azimuthAndzenith(p) for p in event.final_p])
        particles = {
            "primary": np.array(np.zeros(len(event.final_ids)), dtype=bool),  # If this thing is a primary particle
            "e": event.final_e,  # The energy in GeV,  # NOTE: Should this be kinetic energy?
            "pdg_code": event.final_ids,  # The PDG id
            "interaction": np.array([event_type for _ in range(len(event.final_ids))]),  # Should we use ints to define interactions or strings?
            "theta": angles[:, 1],  # It's zenith angle
            "phi": angles[:, 0],  # It's azimuth angle
            "bjorken_x": np.ones(len(event.final_ids)) * -1,  # Bjorken x variable
            "bjorken_y": np.ones(len(event.final_ids)) * -1,  # Bjorken y variable
            "pos_x": np.ones(len(event.final_ids)) * event.event_vertex[0],  # position x (in detector coordinates)
            "pos_y": np.ones(len(event.final_ids)) * event.event_vertex[1],  # position y (in detector coordinates)
            "pos_z": np.ones(len(event.final_ids)) * event.event_vertex[2],  # position z (in detector coordinates)
            "position": np.array([event.event_vertex[:3] for _ in range(len(event.final_ids))]),  # 3d position
            "column_depth": np.ones(len(event.final_ids)) * -1,  # Column depth where the interaction happened
            'custom_info': np.array(['child' for _ in range(len(event.final_ids))])  # Additional information the user can add as a string. # TODO: This should be handed to the propagators
        }
        event_set[index] = particles
        primaries[index] = primary_particle
    return pd.DataFrame.from_dict(event_set, orient='index'), pd.DataFrame.from_dict(primaries, orient='index')

def inject_in_cylinder(primary_set, ## neutrino information
                        prometheus_set, ## child particle information
                        cylinder_radius=500, ## meters
                        cylinder_height = 1000, # meters
                        cylinder_center = (0, 0, 0), # meters
                        ):
    """
    Sample a random point in a cylinder. Then assign the interaction vertex to sampled position and offset the 
    position of child particles by this sampled point.
    IMPORTANT: ASSUMES INTERACTION VERTEX IS AT (0,0,0), which is default by gevgen... also if it wasn't why use this?
    """
    if len(primary_set) != len(prometheus_set):
        raise ValueError('Length of primary set (neutrino informaiton) does not equal length of prometheus set (child particles) !!') 

    n_events = len(primary_set)
    ## Cylinder:
    r = np.sqrt(np.random.uniform(0, 1, n_events)) * cylinder_radius
    theta = np.random.uniform(0, 2*np.pi, n_events) ## theta in circle
    z = np.random.uniform(-cylinder_height/2, cylinder_height/2, n_events)

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
        prometheus_set.loc[i, 'position'] += np.array([np.array(primary_set['position'].iloc[i])[0:3]] * prometheus_set.loc[i, 'position'].shape[0]) ## should time be in here?    
        prometheus_set.loc[i, 'pos_x'] += np.array([np.array(primary_set['pos_x'].iloc[i])] * prometheus_set.loc[i, 'pos_x'].shape[0])
        prometheus_set.loc[i, 'pos_y'] += np.array([np.array(primary_set['pos_y'].iloc[i])] * prometheus_set.loc[i, 'pos_y'].shape[0])
        prometheus_set.loc[i, 'pos_z'] += np.array([np.array(primary_set['pos_z'].iloc[i])] * prometheus_set.loc[i, 'pos_z'].shape[0])

    return primary_set, prometheus_set



def rotate_particles_final(primary_set, prometheus_set):
    """
    Samples a rotation isotopically, then rotates neutrino and child particles
    using Rodrigues' rotation formula to create an isotropic flux
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    IMPORTANT: ASSUMES NEUTRINO UNIT VECTOR DIRECTION IS (0, 0, 1) initially
    """
    if len(primary_set) != len(prometheus_set):
        raise ValueError('Length of primary sets do not match!')
    
    n_events = len(primary_set)
    initial_neutrino = np.array([0, 0, 1]) #### assumes neutrino initial direction is 0,0,1
    
    for i in range(n_events):
        # sample target direction isotropically
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
        
        # Update primary neutrino direction
        primary_set.at[i, 'theta'] = theta
        primary_set.at[i, 'phi'] = phi
        
        # Get the arrays for this event
        theta_array = prometheus_set.loc[i, 'theta'].copy()
        phi_array = prometheus_set.loc[i, 'phi'].copy()
        
        # Rotate each particle and verify changes
        for j in range(len(theta_array)):
            if np.isnan(theta_array[j]) or np.isnan(phi_array[j]):
                continue
                
            # Store original values for verification
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
            
            # print first few particles
            if i < 2 and j < 2:
                print(f"Event {i}, Particle {j}:")
                print(f"  Original: theta={theta_orig:.4f}, phi={phi_orig:.4f}")
                print(f"  Rotated: theta={theta_array[j]:.4f}, phi={phi_array[j]:.4f}")
                print(f"  Changed: {theta_array[j] != theta_orig or phi_array[j] != phi_orig}")
        

        prometheus_set.at[i, 'theta'] = theta_array
        prometheus_set.at[i, 'phi'] = phi_array
        
        # test
        if i == 0:
            print("\nAfter assignment:")
            print(f"  First theta original: {theta_orig:.4f}")
            print(f"  First theta in DataFrame: {prometheus_set.loc[0, 'theta'][0]:.4f}")
            print(f"  Changed in DataFrame: {prometheus_set.loc[0, 'theta'][0] != theta_orig}")
    
    return primary_set, prometheus_set

def rotation_matrix_from_vectors(vec1, vec2):
    """
    calculate rotation matrix that rotates vec1 to vec2.
    buth vectors should be unit vectors.
    """
    # handle special cases
    if np.allclose(vec1, vec2):
        return np.eye(3)  # No rotation needed
    
    # general case - find rotation matrix
    v = np.cross(vec1, vec2)
    s = np.linalg.norm(v)
    c = np.dot(vec1, vec2)
    
    if np.isclose(s, 0):
        # vectors parallel or anti-parallel
        if c > 0:
            return np.eye(3)  # same direction
        else:
            # opposite direction - rotate 180° around any perpendicular axis
            # find a perpendicular vector
            if abs(vec1[0]) < abs(vec1[1]):
                perp = np.array([0, -vec1[2], vec1[1]])
            else:
                perp = np.array([-vec1[2], 0, vec1[0]])
            perp = perp / np.linalg.norm(perp)
            
            # create rotation matrix for 180 degrees around perp
            R = 2 * np.outer(perp, perp) - np.eye(3)
            return R
    
    # regular case - Rodrigues' formula in matrix form
    v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + v_x + v_x.dot(v_x) * (1 - c) / (s * s)
    return R


def final_parser(parsed_events: pd.DataFrame)->pd.DataFrame:
    """ fetches the final states

    Parameters
    ----------
    parsed_events : pd.DataFrame
        The parsed events

    Returns
    -------
    pd.DataFrame
        The inital + final state info
    """
    inital_energies_inj = np.array([event[0][3] for event in parsed_events['event_momenta']])
    inital_momenta_inj = [np.array(event[0][:3]) for event in parsed_events['event_momenta']]
    inital_energies_target = np.array([event[1][3] for event in parsed_events['event_momenta']])
    inital_id_inj = np.array([event[0] for event in parsed_events['event_pdg_id']])
    inital_id_target = np.array([event[1] for event in parsed_events['event_pdg_id']])
    final_ids = np.array([np.where(event == np.array('final'), True, False) for event in parsed_events['event_status']], dtype=object)
    children_ids = np.array([
        event[final_ids[id_event]] for id_event, event in enumerate(parsed_events['event_pdg_id'])
    ], dtype=object)
    children_energy = np.array([
        event[:, 3][final_ids[id_event]] for id_event, event in enumerate(parsed_events['event_momenta'])
    ], dtype=object)
    children_momenta = np.array([
        event[:, :3][final_ids[id_event]] for id_event, event in enumerate(parsed_events['event_momenta'])
    ], dtype=object)
    final_ids = np.array([np.where(event == np.array('final nuclear'), True, False) for event in parsed_events['event_status']], dtype=object)
    children_nuc_ids = np.array([
        event[final_ids[id_event]] for id_event, event in enumerate(parsed_events['event_pdg_id'])
    ], dtype=object)
    children_nuc_energy = np.array([
        event[:, 3][final_ids[id_event]] for id_event, event in enumerate(parsed_events['event_momenta'])
    ], dtype=object)
    dic = {}
    dic['event_descr'] = parsed_events['event_description']
    dic['event_xsec'] = parsed_events['event_xsec']
    dic['event_vertex'] = parsed_events.event_vertex
    dic['init_inj_e'] = inital_energies_inj
    dic['init_inj_p'] = inital_momenta_inj
    dic['init_target_e'] = inital_energies_target
    dic['init_inj_id'] = inital_id_inj
    dic['init_target_id'] = inital_id_target
    dic['final_ids'] = children_ids
    dic['final_e'] = children_energy
    dic['final_p'] = children_momenta
    dic['final_nuc_ids'] = children_nuc_ids
    dic['final_nuc_e'] = children_nuc_energy
    dic['p4'] = np.array([event for event in parsed_events['event_momenta']], dtype='object')
    # for key in ['final_p', 'final_ids', 'final_e', 'final_nuc_ids', 'final_nuc_e']:
    #     if key in dic and isinstance(dic[key], np.ndarray):
    #         dic[key] = pd.Series([x for x in dic[key]], dtype='object')
    return pd.DataFrame.from_dict(dic)

def main():
    with uproot.open('/groups/icecube/jackp/genie_test_outputs/output_gheps/gntp_icecube_numu_100.gtac.root') as file:
        events = file['gRooTracker']
        parsed_events = genie_parser(events)
        final_parsed = final_parser(parsed_events)
    prometheus_set, primary_set = genie2prometheus(final_parsed)
    primary_set_pre_rotation, prometheus_set_pre_rotation = primary_set.copy(), prometheus_set.copy()
    primary_set, prometheus_set = inject_in_cylinder(primary_set, prometheus_set)

    # position arrays to serialized strings, thanks to guy on github
    prometheus_set['position'] = prometheus_set['position'].apply(lambda x: [arr.tolist() for arr in x])
    num_events = 100

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/groups/icecube/jackp/prometheus_genie_cleaned/harvard-prometheus/examples/output"
    primrary_file_path ='/groups/icecube/jackp/prometheus_genie_cleaned/harvard-prometheus/examples/output/genie_events_primary.parquet'
    prometheus_file_path ='/groups/icecube/jackp/prometheus_genie_cleaned/harvard-prometheus/examples/output/genie_events_prometheus.parquet'
    
    prometheus_set.to_parquet(prometheus_file_path)
    primary_set.to_parquet(primrary_file_path)
    print('converted to parquet')

  #  RESOURCE_DIR = f"{'/'.join(prometheus.__path__[0].split('/')[:-1])}/resources/"
    config["injection"]["name"] = "GENIE"
    config["run"]["outfile"] = f"{output_dir}/100_new_lots_events_{timestamp}.parquet"
    config["run"]["nevents"] = num_events
    #   config["run"]["muon_energy"] = 2 
    config["detector"]["geo file"] = f"{RESOURCE_DIR}/geofiles/icecube.geo"
    
    # Ensure 'simulation' key exists in the main config
    config["simulation"] = config.get("simulation", {})
    
    # Set up PPC configuration
    config['photon propagator']['name'] = 'PPC'
    config["photon propagator"]["PPC"] = config["photon propagator"].get("PPC", {})
    config["photon propagator"]["PPC"]["paths"] = config["photon propagator"]["PPC"].get("paths", {})
    config["photon propagator"]["PPC"]["paths"]["ppc_tmpdir"] = "./ppc_tmpdir"
    config["photon propagator"]["PPC"]["paths"]["ppctables"] = f"{RESOURCE_DIR}/PPC_tables/spx"
    simset=1
    config["photon propagator"]["PPC"]["paths"]["ppc_tmpfile"] = "ppc_tmp"+str(simset)
    config["photon propagator"]["PPC"]["simulation"]["supress_output"] = False
    print(config["photon propagator"]["PPC"])

    logger.info(f"Using photon propagator: {config['photon propagator']['name']}")
    logger.info(f"PPC tables path: {config['photon propagator']['PPC']['paths']['ppctables']}")

    config["injection"]["GENIE"] = config["injection"].get("GENIE", {})
    config["injection"]["GENIE"]["paths"] = config["injection"]["GENIE"].get("paths", {})
   # config["injection"]["GENIE"]["paths"]["primary file"] = h5_file_path
    config["injection"]["GENIE"]["inject"] = False
    config["injection"]["GENIE"]["simulation"] = {}

    # tmp_dir = "./ppc_tmpdir"
    # if os.path.exists(tmp_dir):
    #     shutil.rmtree(tmp_dir)

    logger.info("Starting Prometheus simulation with PPC")
    prometheus_start = time.time()
    p = Prometheus(config, primary_set_parquet_path=primrary_file_path, prometheus_set_parquet_path=prometheus_file_path)
    p.sim()
    print('finished without catastrophic error')
    return

if __name__ == "__main__":
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("Launching simulation")
    main()
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
