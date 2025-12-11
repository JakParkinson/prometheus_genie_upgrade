


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
from gevgen_injection import run_gevgen, convert_to_gst, convert_to_gtrac
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
    parser.add_argument('--num_events', type=int, 
                        default='10',
                        help='Number of events')
    args = parser.parse_args()
    
    # Use the arguments
    simset = args.simset
    num_events = args.num_events
    
    print(f"Using simset: {simset}, num_events {num_events}")

    processing_start_time = time.time()

    logger.info("Starting GENIE simulation")
    genie_start = time.time()
    ghep_file = run_gevgen(num_events, OUTPUT_DIR)
    gtrac_file = convert_to_gtrac(ghep_file, OUTPUT_DIR)
    
    genie_time = time.time() - genie_start
    logger.info(f"GENIE simulation completed in {genie_time:.2f} seconds")
    prometheus_set, primary_set = parse_and_convert_genie(gtrac_file)
    processing_end_time = time.time()
    print(f"File processing and conversion completed in {processing_end_time - processing_start_time:.2f} seconds")


    primrary_file_path = f'{OUTPUT_DIR}/genie_events_primary_new_test_simset_{simset}.parquet'
    prometheus_file_path = f'{OUTPUT_DIR}/genie_events_prometheus_new_test_simset_{simset}.parquet'

    primary_set, prometheus_set = inject_particles_in_cylinder(
        primary_set,  # neutrino information
        prometheus_set,  # child particle information
        cylinder_radius=100.00,  # meters
        cylinder_height=400.00,  # meters
        cylinder_center=(45.34, -58.61, -2282.0),  # meters
        detector_offset = (10.12, -8.56, -2005.37) ## Upgrade Specific Offset!
    ) ## coordinates based on a cylinder in the geometric center of  upgrade
    ## upgrade geometric center is: 45.33571428571429, -58.607142857142854, -2282.0
    ## upgrade offset is 10.12236383    -8.56100121 -2005.37453744
    ## The offset is something specific to a detector that prometheus calculates.
    #  I added a print statement that prints the detector offset. If you are not using 'icecube_upgrade_new.geo', then you need to use the detector specific offset
       # cylinder_center=(35.1, -50.05, -276.63),  # meters
    

    primary_set, prometheus_set = rotate_particles_final(primary_set, prometheus_set)

    # Timing for serialization and saving
    save_start_time = time.time()
    # position arrays to serialized strings
    prometheus_set['position'] = prometheus_set['position'].apply(lambda x: [arr.tolist() for arr in x])
    num_events = len(prometheus_set)
    print(f"Processing {num_events} events")
    prometheus_set.to_parquet(prometheus_file_path)
    primary_set.to_parquet(primrary_file_path)
    save_end_time = time.time()
    print(f"Serialization and saving completed in {save_end_time - save_start_time:.2f} seconds")

    ## GENIE stuff:
    config["injection"]["name"] = "GENIE"
    config["run"]["outfile"] = f"{OUTPUT_DIR}/{num_events}events_simset_{simset}.parquet"
    config["run"]["nevents"] = num_events
    #config["injection"]["GENIE"] = config["injection"].get("GENIE", {})
   # config["injection"]["GENIE"]["paths"] = config["injection"]["GENIE"].get("paths", {})
    config["injection"]["GENIE"]["inject"] = False ## we aren't injecting using prometheus, we are using an injection file
    config["injection"]["GENIE"]["simulation"] = {}
    ## geofile:
   # config["detector"]["geo file"] = f"{RESOURCE_DIR}/geofiles/icecube_upgrade_new.geo"
    #config["detector"]["geo file"] = f"{RESOURCE_DIR}/geofiles/icecube_upgrade_new.geo"
    config["detector"]["geo file"] = f"{RESOURCE_DIR}/geofiles/super_simple_test.geo"
    
    
    ## ppc configuration:
    config['photon propagator']['name'] = 'PPC_UPGRADE'
    #config["photon propagator"]["PPC"] = config["photon propagator"].get("PPC", {})
   # config["photon propagator"]["PPC"]["paths"] = config["photon propagator"]["PPC"].get("paths", {})
    config["photon propagator"]["PPC_UPGRADE"]["paths"]["ppc_tmpdir"] = "./ppc_tmpdir" + str(simset)
    
    config["photon propagator"]["PPC_UPGRADE"]["paths"]["ppc_tmpfile"] = "ppc_tmp"+str(simset)
    config["photon propagator"]["PPC_UPGRADE"]["simulation"]["supress_output"] = False ## for printing!

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