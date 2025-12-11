#!/usr/bin/env python3

import os
import subprocess
import logging
from datetime import datetime

import time
import csv


import h5py
import numpy as np
import pandas as pd



import pandas as pd
import numpy as np
import uproot
from schema import Schema, And, Use, Optional, SchemaError

import traceback

import sys
import os
import shutil
import logging
sys.path.append('../')

from prometheus import Prometheus, config
import prometheus
import gc

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# logging.getLogger('prometheus.photon_propagation.ppc_photon_propagator').setLevel(logging.DEBUG)
# logging.getLogger('prometheus.photon_propagation.utils').setLevel(logging.DEBUG)
# logging.getLogger('prometheus.injection.injection.genie_injection').setLevel(logging.DEBUG)



def run_gevgen(num_events, output_dir):
    """Run GEVGEN with specified parameters"""
    ghep_file = os.path.join(output_dir, f"gntp_icecube_{num_events}_events.ghep.root")

    genie_path = os.getenv("GENIE")  # Get the GENIE environment variable

    if not genie_path:
        raise EnvironmentError("GENIE environment variable is not set.")


    cmd = [
        os.path.join(genie_path, "bin", "gevgen"), 
        "-n", str(num_events),
        "-p", "14", ## muon neutrino
        "-t", "1000080160[0.888],1000010010[0.112]", ## H2O
        "-e", "1,100", ## in GeV
        "-f", "x^-1", ## Power Law
        "--seed", "12345",
        "--cross-sections", "/groups/icecube/jackp/genie-3.4.2/ice_numu_cross_sections.xml", ## combined H and O16 cross section filex into 1
        "--event-generator-list", "Default",
        "--event-record-print-level", "3",
        "-o", ghep_file,
        "--debug-flags", "PhysModels,EventGen,NucleonDecay",
        "--tune", "G18_02a_00_000"
    ]
    
    logger.info(f"Running GEVGEN with {num_events} events")
    print("cmd: ", cmd)
    logger.info("genie command: %s", cmd)

    #subprocess.run(cmd, check=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info(f"GEVGEN completed. Output: {ghep_file}")
    return ghep_file

def convert_to_gst(ghep_file, output_dir):
    """Convert GHEP file to GST format"""
    gst_file = os.path.join(output_dir, "ice_cube_sim_summary.root")
    genie_path = os.getenv("GENIE")

    if not genie_path:
        raise EnvironmentError("GENIE environment variable is not set.")

    cmd = [
        os.path.join(genie_path, "bin", "gntpc"),  # Use the actual path to gntpc
        "-i", ghep_file,
        "-f", "gst",
        "-o", gst_file
    ]
    
    logger.info("Converting to GST format")
    subprocess.run(cmd, check=True)
    logger.info(f"GST conversion completed. Output: {gst_file}")
    return gst_file

def convert_to_gtrac(ghep_file, output_dir):
    """Convert GHEP file to GTRAC format using rootracker"""
    gtrac_file = os.path.join(output_dir, "ice_cube_sim.gtrac.root")
    genie_path = os.getenv("GENIE")  # Fetch the GENIE environment variable

    if not genie_path:
        raise EnvironmentError("GENIE environment variable is not set.")

    cmd = [
        os.path.join(genie_path, "bin", "gntpc"),  # Use the actual path to gntpc
        "-i", ghep_file,
        "-f", "rootracker",
        "-o", gtrac_file
    ]

    logger.info("Converting to GTRAC format using rootracker")
    subprocess.run(cmd, check=True)
    logger.info(f"GTRAC conversion completed. Output: {gtrac_file}")
    return gtrac_file





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
        The azimuth and zenith angles in radians.
    """
    azimuth = angle(p[:2], np.array([0, 1]))
    zenith = angle(p[1:], np.array([0., 1]))
    return azimuth, zenith
def distribute_in_icecube_cylinder(genie_data, 
                                  cylinder_radius=500.0,  # meters
                                  cylinder_height=1000.0, # meters
                                  cylinder_center=(0, 0, 0),
                                  randomize_direction=True):
    """
    Distribute neutrino interaction vertices within an IceCube-like cylinder
    and adjust momentum directions if requested. Child particles maintain
    their relative positions from the vertex.
    """
    import numpy as np
    
    # copy
    modified_data = genie_data.copy()
    n_events = len(modified_data)
    
    # Method is: select a 
    r = np.sqrt(np.random.uniform(0, 1, n_events)) * cylinder_radius  # sqrt for uniform sampling inside circle
    theta = np.random.uniform(0, 2*np.pi, n_events) ## the azimuth angle 
    z = np.random.uniform(-cylinder_height/2, cylinder_height/2, n_events) # the length of cylinder
    
    # to cartesian
    x = r * np.cos(theta) + cylinder_center[0]
    y = r * np.sin(theta) + cylinder_center[1]
    z = z + cylinder_center[2]
    
    # update vertex positions
    new_vertices = []
    for i in range(n_events):
        # create new vertex (x, y, z, t) with t=0
        new_vertices.append((x[i], y[i], z[i], 0.0))
    
    modified_data['event_vertex'] = new_vertices
    print("new event vertices: ", new_vertices)
    
    # randomize directions if requested
    if randomize_direction:
        new_momenta = []
        new_children_momenta = []
        
        for i in range(n_events):
            # Get original momentum and its magnitude
            p_orig = modified_data['init_inj_p'].iloc[i]
            p_mag = np.sqrt(p_orig[0]**2 + p_orig[1]**2 + p_orig[2]**2)
            
            # Sample a random rotation angle (0 to π)
            angle = np.random.uniform(0, np.pi)
            
            # Sample random rotation axis uniformly on sphere
            phi = np.random.uniform(0, 2*np.pi)
            cos_alpha = np.random.uniform(-1, 1)
            sin_alpha = np.sqrt(1 - cos_alpha**2)
            rotation_axis = np.array([
                sin_alpha * np.cos(phi),
                sin_alpha * np.sin(phi),
                cos_alpha
            ])
            
            # Ensure the rotation axis is normalized
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            # Create rotation matrix using Rodrigues formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rotation_matrix = np.identity(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
            
            # Apply rotation to neutrino momentum
            p_vector = np.array(p_orig)
            new_p = rotation_matrix @ p_vector
            new_momenta.append(tuple(new_p))
            
            # Get child momenta
            event_momenta = modified_data['final_p'].iloc[i]
            if len(event_momenta) == 0:  # append empty tuple if no child particles
                new_children_momenta.append(())
                continue
            
            # Apply same rotation to each child momentum
            rotated_child_momenta = []
            for child_p in event_momenta:
                p_vector = np.array(child_p)
                rotated_p = rotation_matrix @ p_vector
                rotated_child_momenta.append(tuple(rotated_p))
            
            new_children_momenta.append(tuple(rotated_child_momenta))
        
        modified_data['init_inj_p'] = new_momenta
        print("new init p: ", new_momenta)
        
        modified_data['final_p'] = new_children_momenta
        print("new final_p: ", new_children_momenta)
    
    return modified_data

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
    print("event description: ", dic['event_description'])
    dic['event_id'] = events['EvtNum'].array(library="np")  # Number of the event
    print("event id: ", dic['event_id'])
    dic['event_children'] = events['StdHepN'].array(library="np")  # NUmber of the generated sub-particles
    print("evebt childrin: ", dic['event_children'])
    dic['event_prob'] = events['EvtProb'].array(library="np")  # Probability of the event
    print("event_prob: ", dic['event_prob'])
    dic['event_xsec'] = events['EvtXSec'].array(library="np")  # Total xsec of the event
    print("event_xsec: ", dic['event_xsec'])
    dic['event_pdg_id'] = events['StdHepPdg'].array(library="np")  # PDG ids of all produced particles
    print("event_pdg_id: ", dic['event_pdg_id'])
    dic['event_momenta'] = events['StdHepP4'].array(library="np")  # Momenta of the particles
    tmp = events['EvtVtx'].array(library="np")  # Position of the particles
    
    dic['event_vertex'] = [np.array(vtx) for vtx in tmp]
    print("event vertix: ", dic['event_vertex'])
    dic['event_coords'] = events['StdHepX4'].array(library="np")  # Position of the particles
    print("event_coords ", dic["event_coords"])
    dic['event_weight'] = events['EvtWght'].array(library='np')  # Weight of the events
    print("event_weight: ", dic['event_weight'])
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

def final_parser_problem(parsed_events: pd.DataFrame)->pd.DataFrame:
    inital_energies_inj = np.array([event[0][3] for event in parsed_events['event_momenta']])
    print("initial energies inj: ", inital_energies_inj)
    inital_momenta_inj = [tuple(event[0][:3]) for event in parsed_events['event_momenta']]  # Convert to tuple
    print("initial momenta inj: ", inital_momenta_inj)
    inital_energies_target = np.array([event[1][3] for event in parsed_events['event_momenta']])
    print("initial energies target: ", inital_energies_target)
    inital_id_inj = np.array([event[0] for event in parsed_events['event_pdg_id']])
    print("initial id inj: ", inital_id_inj)
    inital_id_target = np.array([event[1] for event in parsed_events['event_pdg_id']])
    print("initial_id_target: ", inital_id_target)
    final_ids = np.array([np.where(event == np.array('final'), True, False) for event in parsed_events['event_status']], dtype=object)
    children_ids = [tuple(event[final_ids[id_event]]) for id_event, event in enumerate(parsed_events['event_pdg_id'])]  # Convert to tuple
    children_energy = [tuple(event[:, 3][final_ids[id_event]]) for id_event, event in enumerate(parsed_events['event_momenta'])]  # Convert to tuple
    children_momenta = [tuple(tuple(p) for p in event[:, :3][final_ids[id_event]]) for id_event, event in enumerate(parsed_events['event_momenta'])]  # Convert to tuple of tuples
    final_ids = np.array([np.where(event == np.array('final nuclear'), True, False) for event in parsed_events['event_status']], dtype=object)
    children_nuc_ids = [tuple(event[final_ids[id_event]]) for id_event, event in enumerate(parsed_events['event_pdg_id'])]  # Convert to tuple
    children_nuc_energy = [tuple(event[:, 3][final_ids[id_event]]) for id_event, event in enumerate(parsed_events['event_momenta'])]  # Convert to tuple

    dic = {}
    dic['event_descr'] = parsed_events['event_description']
    dic['event_xsec'] = parsed_events['event_xsec']
    dic['event_vertex'] = [tuple(v) for v in parsed_events.event_vertex]  # Convert to tuple
    print('event_vertex: ', dic['event_vertex'])
    dic['init_inj_e'] = inital_energies_inj
    dic['init_inj_p'] = inital_momenta_inj
    print('init_inj_p: ', inital_momenta_inj)
    dic['init_target_e'] = inital_energies_target
    dic['init_inj_id'] = inital_id_inj
    dic['init_target_id'] = inital_id_target
    dic['final_ids'] = children_ids
    dic['final_e'] = children_energy
    dic['final_p'] = children_momenta
    print('children momenta: ', children_momenta)
    dic['final_nuc_ids'] = children_nuc_ids
    dic['final_nuc_e'] = children_nuc_energy
    dic['p4'] = [tuple(tuple(p) for p in event) for event in parsed_events['event_momenta']]  # Convert to tuple of tuples

    return pd.DataFrame.from_dict(dic)





all_fields = [
            "injection_energy",
            "injection_type",
            "injection_interaction_type",
            "injection_zenith",
            "injection_azimuth",
            "injection_bjorkenx",
            "injection_bjorkeny",
            "injection_position_x",
            "injection_position_y",
            "injection_position_z",
            "injection_column_depth",
            "primary_particle_1_type",
            "primary_particle_1_position_x",
            "primary_particle_1_position_y",
            "primary_particle_1_position_z",
            "primary_particle_1_direction_theta",
            "primary_particle_1_direction_phi",
            "primary_particle_1_energy",
            "primary_particle_2_type",
            "primary_particle_2_position_x",
            "primary_particle_2_position_y",
            "primary_particle_2_position_z",
            "primary_particle_2_direction_theta",
            "primary_particle_2_direction_phi",
            "primary_particle_2_energy",
            "total_energy",
        ]

def schema_check(std_schema: Schema, to_check)->bool:
    """ quick and dirty function to validate formatting
    """
    try:
        std_schema.validate(to_check)
        return True
    except SchemaError:
        return False
# inj_particle_scheme = Schema({
#         "primary": And(Use(bool)),  # If this thing is a primary particle
#         "energy": And(Use(float)),  # The energy in GeV,  # NOTE: Should this be kinetic energy?
#         "PDG": And(Use(int)),  # The PDG id
#         "interaction": And(Use(str)),  # Should we use ints to define interactions or strings?
#         "theta": And(Use(float)),  # It's zenith angle
#         "phi": And(Use(float)),  # It's azimuth angle
#         "bjorken_x": And(Use(float)),  # Bjorken x variable
#         "bjorken_y": And(Use(float)),  # Bjorken y variable
#         "pos_x": And(Use(float)),  # position x (in detector coordinates)
#         "pos_y": And(Use(float)),  # position y (in detector coordinates)
#         "pos_z": And(Use(float)),  # position z (in detector coordinates)
#         "column_depth": And(Use(float)),  # Column depth where the interaction happened
#         Optional('custom_info'): And(Use(str))  # Additional information the user can add as a string. # TODO: This should be handed to the propagators
#     })

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



def genie_to_h5_updated(genie_data, output_file, max_final_particles=20):
    with h5py.File(output_file, 'w') as f:
        injector = f.create_group('RangedInjector0')
        
        particle_dtype = np.dtype({
            'names': ['initial', 'ParticleType', 'Position', 'Direction', 'Energy'],
            'formats': ['u1', '<i4', ('<f8', (3,)), ('<f8', (2,)), '<f8'],
            'offsets': [0, 4, 8, 32, 48],
            'itemsize': 64
        })
        
        properties_dtype = np.dtype({
            'names': ['totalEnergy', 'zenith', 'azimuth', 'finalStateX', 'finalStateY'] + 
                     [f'finalType{i}' for i in range(1, max_final_particles+1)] + 
                     ['initialType', 'x', 'y', 'z', 'totalColumnDepth', 'numFinalParticles'],
            'formats': ['<f8', '<f8', '<f8', '<f8', '<f8'] + 
                        ['<i4'] * max_final_particles + 
                        ['<i4', '<f8', '<f8', '<f8', '<f8', '<i4'],
            'offsets': list(range(0, 40, 8)) + list(range(40, 40 + 4*max_final_particles, 4)) + 
                       [40 + 4*max_final_particles + i*8 for i in range(6)],
            'itemsize': 48 + 4*max_final_particles + 48
        })
        
        n_events = len(genie_data)
        
        # Initial particles
        initial_data = np.zeros(n_events, dtype=particle_dtype)
        initial_data['initial'] = 1
        initial_data['ParticleType'] = genie_data['init_inj_id'].astype(int)
        initial_data['Position'] = np.array(genie_data['event_vertex'].tolist())[:, :3]
        initial_data['Direction'][:, 0] = np.arctan2(genie_data['init_inj_p'].apply(lambda x: x[1]), genie_data['init_inj_p'].apply(lambda x: x[0]))
        initial_data['Direction'][:, 1] = np.arccos(genie_data['init_inj_p'].apply(lambda x: x[2] / np.linalg.norm(x)))
        initial_data['Energy'] = genie_data['init_inj_e']
        injector.create_dataset('initial', data=initial_data)
        
        # Final particles
        for i in range(1, max_final_particles + 1):
            final_data = np.zeros(n_events, dtype=particle_dtype)
            final_data['initial'] = 0
            for j, (ids, energies, momenta) in enumerate(zip(genie_data['final_ids'], genie_data['final_e'], genie_data['final_p'])):
                if i <= len(ids):
                    final_data[j]['ParticleType'] = ids[i-1]
                    final_data[j]['Position'] = genie_data['event_vertex'].iloc[j][:3]
                    final_data[j]['Direction'][0] = np.arctan2(momenta[i-1][1], momenta[i-1][0])
                    final_data[j]['Direction'][1] = np.arccos(momenta[i-1][2] / np.linalg.norm(momenta[i-1]))
                    final_data[j]['Energy'] = energies[i-1]
            injector.create_dataset(f'final_{i}', data=final_data)
        
        # Properties
        ### todo : update to match final_data i think?? like the finalsStateX and finalStateY...
        properties_data = np.zeros(n_events, dtype=properties_dtype)
        properties_data['totalEnergy'] = genie_data['init_inj_e'] + genie_data['init_target_e']
        print('totaLEnergy: ', properties_data['totalEnergy'])
        properties_data['zenith'] = np.arccos(genie_data['init_inj_p'].apply(lambda x: x[2] / np.linalg.norm(x)))
        print('zenith: ', properties_data['zenith'])
        properties_data['azimuth'] = np.arctan2(genie_data['init_inj_p'].apply(lambda x: x[1]), genie_data['init_inj_p'].apply(lambda x: x[0]))
        print('azimuth: ', properties_data['azimuth'])
        properties_data['finalStateX'] = genie_data['final_e'].apply(lambda x: x[0]/sum(x) if len(x) > 0 else 0)  ## not sure , need to revisit, also not sure if it really matters for GENIE
        properties_data['finalStateY'] = genie_data['final_e'].apply(lambda x: x[1]/sum(x) if len(x) > 1 else 0)
        for i in range(1, max_final_particles + 1):
            properties_data[f'finalType{i}'] = genie_data['final_ids'].apply(lambda x: x[i-1] if len(x) >= i else 0).astype(int)
        properties_data['initialType'] = genie_data['init_inj_id'].astype(int)
        print('final state X ', properties_data['finalStateX'])
        print('final state Y ', properties_data['finalStateY'])
        properties_data['x'] = genie_data['event_vertex'].apply(lambda x: x[0])
        properties_data['y'] = genie_data['event_vertex'].apply(lambda x: x[1])
        properties_data['z'] = genie_data['event_vertex'].apply(lambda x: x[2])
        properties_data['totalColumnDepth'] = np.ones(len(genie_data['event_xsec']))*-1
        print("totalColumnDepth: ", properties_data['totalColumnDepth'])
        properties_data['numFinalParticles'] = genie_data['final_ids'].apply(len)
        print("numFinalParticles: ", properties_data['numFinalParticles'])
        
        injector.create_dataset('properties', data=properties_data)


def run_benchmark(num_events):
    start_time = time.time()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/groups/icecube/jackp/icecube_genie_sims/icecube_benchmark_{num_events}_events_{timestamp}"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

        logger.info("Starting GENIE simulation")
        genie_start = time.time()
        ghep_file = run_gevgen(num_events, output_dir)
        gst_file = convert_to_gst(ghep_file, output_dir)
        gtrac_file = convert_to_gtrac(ghep_file, output_dir)
        genie_time = time.time() - genie_start
        logger.info(f"GENIE simulation completed in {genie_time:.2f} seconds")

        logger.info("Starting file conversion")
        file_conversion_start = time.time()
        with uproot.open(gtrac_file) as file:
            events = file['gRooTracker']
            parsed_events = genie_parser(events)
            final_parsed = final_parser_problem(parsed_events)
        

            final_parsed = distribute_in_icecube_cylinder(
                final_parsed,
                cylinder_radius=500.0,  # 500m radius
                cylinder_height=1000.0  # 1000m height
            )

        h5_file_path = f"{output_dir}/genie_output_{num_events}_events_{timestamp}.h5" 
        genie_to_h5_updated(final_parsed, h5_file_path)
        file_conversion_time = time.time() - file_conversion_start
        logger.info(f"File conversion completed in {file_conversion_time:.2f} seconds")

        logger.info("Setting up Prometheus configuration")
        RESOURCE_DIR = f"{'/'.join(prometheus.__path__[0].split('/')[:-1])}/resources/"
        config["injection"]["name"] = "GENIE"
        config["run"]["outfile"] = f"{output_dir}/1000_new_lots_events_{timestamp}.parquet"
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
        config["injection"]["GENIE"]["paths"]["injection file"] = h5_file_path
        config["injection"]["GENIE"]["inject"] = False
        config["injection"]["GENIE"]["simulation"] = {}

        # tmp_dir = "./ppc_tmpdir"
        # if os.path.exists(tmp_dir):
        #     shutil.rmtree(tmp_dir)

        logger.info("Starting Prometheus simulation with PPC")
        prometheus_start = time.time()
        p = Prometheus(config, external_h5_file=h5_file_path)
        p.sim()
        prometheus_time = time.time() - prometheus_start
        logger.info(f"Prometheus simulation completed in {prometheus_time:.2f} seconds")



    except Exception as e:
        logger.error(f"An error occurred during benchmark: {str(e)}")
        logger.error(traceback.format_exc())
        return None

    total_time = time.time() - start_time
    logger.info(f"Total benchmark time: {total_time:.2f} seconds")






def main():
    event_counts = [2]  ## problem with 1 event... somethign with root file indexing ?? yuck
    results = []


    logger.info("starting simulation")
    
    try:
        for num_events in event_counts:
            logger.info(f"Running benchmark with {num_events} events, PPC")
            # try:
                
            run_benchmark(num_events)
            #     if result:
            #         logger.info("did simulation")
            #         results.append(result)
            #         logger.info(f"Benchmark completed successfully for {num_events} events, PPC")
                    
            #         # Append the result to the CSV file immediately
            #         with open(csv_file, 'a', newline='') as file:
            #             writer = csv.DictWriter(file, fieldnames=result.keys())
            #             if file.tell() == 0:  # If file is empty, write header
            #                 writer.writeheader()
            #             writer.writerow(result)
                    
            #         logger.info(f"Result appended to {csv_file}")
            #     else:
            #         raise RuntimeError(f"Benchmark failed for {num_events} events, PPC")
            # except Exception as e:
            #     logger.error(f"Error in benchmark for {num_events} events, PPC: {str(e)}")
            #     logger.error(traceback.format_exc())
            #     break  # Stop the benchmarking if an error occurs

        # if results:
        #     logger.info(f"All benchmarks completed successfully. Total benchmarks: {len(results)}")
        #     logger.info(f"All benchmark results saved to {csv_file}")
        # else:
        #     logger.error("No benchmark results were collected.")
    
    except Exception as e:
        logger.error(f"An error occurred during the benchmarking process: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()