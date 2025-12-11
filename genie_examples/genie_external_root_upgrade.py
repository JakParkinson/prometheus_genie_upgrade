import pandas as pd
import numpy as np
import time
import sys
import argparse
import os
import logging
from datetime import datetime

sys.path.append('../')
from prometheus import Prometheus, config
import prometheus
import gc

# Import our utility functions
from genie_parser_injection import parse_and_convert_genie

from inject_in_cylinder import inject_particles_in_cylinder
from rotate_particles import rotate_particles_final
from aggregate_hadronic_shower import group_hadronic_showers

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RESOURCE_DIR = f"{'/'.join(prometheus.__path__[0].split('/')[:-1])}/resources/"
OUTPUT_DIR = f"{'/'.join(prometheus.__path__[0].split('/')[:-1])}/genie_examples/output"

def main():
    # Start timing
    start_time = time.time()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run GENIE simulation with Prometheus')
    parser.add_argument('--simset', type=int, default=1, 
                        help='Simulation set number (default: 1)')
    parser.add_argument('--nseed', type=int, default=1, 
                        help='genie rnd seed')
    parser.add_argument('--rootfile', type=str, 
                        default='/groups/icecube/jackp/genie_test_outputs/output_gheps/gntp_icecube_numu_100.gtac.root',
                        help='Path to the root file')
    args = parser.parse_args()
    
    # Use the arguments
    simset = args.simset
    root_file_path = args.rootfile
    seed_num = args.nseed
    
    print(f"Using simset: {simset}")
    print(f"Using root file: {root_file_path}")
    
    # Timing for file processing and conversion in one step
    processing_start_time = time.time()
    prometheus_set, primary_set = parse_and_convert_genie(root_file_path)
    processing_end_time = time.time()
    print(f"File processing and conversion completed in {processing_end_time - processing_start_time:.2f} seconds")
    num_events = len(prometheus_set)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   # output_dir = f"/groups/icecube/jackp/prometheus_genie_cleaned/harvard-prometheus/examples/output"
    primrary_file_path = f'{OUTPUT_DIR}/genie_events_primary_new_test_1_to_100gev_simset_{simset}_nevents_{num_events}_seed_{seed_num}_upgrade_internal.parquet'
    prometheus_file_path = f'{OUTPUT_DIR}/genie_events_prometheus_new_test_1_to_100gev_{simset}_nevents_{num_events}_seed_{seed_num}_upgrade_internal.parquet'

    # inject into cylinder:
    primary_set, prometheus_set = inject_particles_in_cylinder(
        primary_set,  # neutrino information
        prometheus_set,  # child particle information
        cylinder_radius=100.00,  # meters
        cylinder_height=350.00,  # meters
        cylinder_center=(44.37, -36.95, -308.00),  # meters
    ) ## coordinates based on a cylinder in the middle of deepcore
    ## offset is [5.87082946, -2.51860853, -1971.9757655]

    primary_set, prometheus_set = rotate_particles_final(primary_set, prometheus_set)


    # Timing for serialization and saving
    save_start_time = time.time()
    # position arrays to serialized strings
    prometheus_set['position'] = prometheus_set['position'].apply(lambda x: [arr.tolist() for arr in x])
    
    print(f"Processing {num_events} events")
    prometheus_set.to_parquet(prometheus_file_path)
    primary_set.to_parquet(primrary_file_path)
    save_end_time = time.time()
    print(f"Serialization and saving completed in {save_end_time - save_start_time:.2f} seconds")

    ## GENIE stuff:
    config["injection"]["name"] = "GENIE"
    config["run"]["outfile"] = f"{OUTPUT_DIR}/final_out_{num_events}_events_1_to_100gev_simset_{simset}_seed_{seed_num}_upgrade_OMSim_{timestamp}_2D_muon_loss.parquet"
    config["run"]["nevents"] = num_events
    config["injection"]["GENIE"] = config["injection"].get("GENIE", {})
    config["injection"]["GENIE"]["paths"] = config["injection"]["GENIE"].get("paths", {})
    config["injection"]["GENIE"]["inject"] = False ## we aren't injecting using prometheus, we are using an injection file
    config["injection"]["GENIE"]["simulation"] = {}
    ## geofile:
    #config["detector"]["geo file"] = f"{RESOURCE_DIR}/geofiles/icecube_upgrade_updated.geo"
    config["detector"]["geo file"] = f"{RESOURCE_DIR}/geofiles/icecube.geo"
    
    ## ppc configuration:
    config['photon propagator']['name'] = 'PPC_UPGRADE'
    config["photon propagator"]["PPC_UPGRADE"]["paths"]["ppc_tmpdir"] = "./ppc_tmpdir" + str(simset)
    config["photon propagator"]["PPC_UPGRADE"]["paths"]["ppc_tmpfile"] = "ppc_tmp"+str(simset)
    config["photon propagator"]["PPC_UPGRADE"]["simulation"]["supress_output"] = True ## for printing!
    # Timing for Prometheus simulation
    sim_start_time = time.time()
    p = Prometheus(config, primary_set_parquet_path=primrary_file_path, prometheus_set_parquet_path=prometheus_file_path)
    p.sim()
    sim_end_time = time.time()
    print(f"Prometheus simulation completed in {sim_end_time - sim_start_time:.2f} seconds")
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print('Finished without catastrophic error')
    return


if __name__ == "__main__":
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("Launching simulation")
    main()
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
