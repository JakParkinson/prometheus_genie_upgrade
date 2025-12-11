#### using the genie_parser.py adn genie_test.ipynb

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
import argparse

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RESOURCE_DIR = f"{'/'.join(prometheus.__path__[0].split('/')[:-1])}/resources/"
OUTPUT_DIR = f"{'/'.join(prometheus.__path__[0].split('/')[:-1])}/examples/output/"



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



### below handles the config and such:
def main():
    start_time = time.time()
    #### default simset is 1, default rootfile is a 100 event file
    parser = argparse.ArgumentParser(description='Run GENIE simulation with Prometheus')
    parser.add_argument('--simset', type=int, default=1, 
                        help='Simulation set number (default: 1)')
    parser.add_argument('--rootfile', type=str, 
                        default='/groups/icecube/jackp/genie_test_outputs/output_gheps/gntp_icecube_numu_100.gtac.root',
                        help='Path to the root file')
    args = parser.parse_args()

    simset = args.simset
    root_file_path = args.rootfile
    with uproot.open(root_file_path) as file:
        events = file['gRooTracker']
        parsed_events = genie_parser(events)
        final_parsed = final_parser(parsed_events)

    prometheus_set, primary_set = genie2prometheus(final_parsed)


    

    ### absolute path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  #  output_dir = f"/groups/icecube/jackp/prometheus_genie_cleaned/harvard-prometheus/examples/output"
    primrary_file_path =f'{OUTPUT_DIR}/genie_events_primary_test.parquet'
    prometheus_file_path =f'{OUTPUT_DIR}/genie_events_prometheus_test.parquet'

    # position arrays to serialized strings, thanks to guy on github
    prometheus_set['position'] = prometheus_set['position'].apply(lambda x: [arr.tolist() for arr in x])
    num_events = len(prometheus_set)
  #  print(len(prometheus_set))
    prometheus_set.to_parquet(prometheus_file_path)
    primary_set.to_parquet(primrary_file_path)

    ## GENIE stuff:
    config["injection"]["name"] = "GENIE"
    config["run"]["outfile"] = f"{OUTPUT_DIR}/100_new_lots_events_{timestamp}.parquet"
    config["run"]["nevents"] = num_events
    config["injection"]["GENIE"] = config["injection"].get("GENIE", {})
    config["injection"]["GENIE"]["paths"] = config["injection"]["GENIE"].get("paths", {})
    config["injection"]["GENIE"]["inject"] = False ## we aren't injecting using prometheus, we are using an injection!!!
    config["injection"]["GENIE"]["simulation"] = {}
    ## geofile:
    config["detector"]["geo file"] = f"{RESOURCE_DIR}/geofiles/icecube.geo"
    ## ppc configuration:
    config['photon propagator']['name'] = 'PPC'
    config["photon propagator"]["PPC"] = config["photon propagator"].get("PPC", {})
    config["photon propagator"]["PPC"]["paths"] = config["photon propagator"]["PPC"].get("paths", {})
    config["photon propagator"]["PPC"]["paths"]["ppc_tmpdir"] = "./ppc_tmpdir"
    config["photon propagator"]["PPC"]["paths"]["ppctables"] = f"{RESOURCE_DIR}/PPC_tables/spx" ## spx for test
    
    config["photon propagator"]["PPC"]["paths"]["ppc_tmpfile"] = "ppc_tmp"+str(simset)
    config["photon propagator"]["PPC"]["simulation"]["supress_output"] = False ## for printing!

    p = Prometheus(config, primary_set_parquet_path=primrary_file_path, prometheus_set_parquet_path=prometheus_file_path)
    p.sim()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print('Finished without catastrophic error :D')
    return


if __name__ == "__main__":
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("Launching simulation")
    main()
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
