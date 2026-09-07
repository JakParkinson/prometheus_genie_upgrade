import numpy as np
import time
import sys
import argparse
import logging
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_DIR = EXAMPLE_DIR.parent

# Make the repository's prometheus package importable
sys.path.insert(0, str(REPO_DIR))

from prometheus import Prometheus, config

from genie_parser_injection import parse_and_convert_genie
from inject_in_cylinder import inject_particles_in_cylinder
from rotate_particles import rotate_particles_final

RESOURCE_DIR = REPO_DIR / "resources"
OUTPUT_DIR = EXAMPLE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    # Start timing
    start_time = time.time()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run GENIE simulation with Prometheus')
    parser.add_argument('--simset', type=int, default=1, 
                        help='Simulation set number (default: 1)')
    parser.add_argument('--nseed', type=int, default=1, 
                        help='Random seed (default: 1)')
    parser.add_argument('--rootfile', type=str, 
                        default=str(EXAMPLE_DIR / "input" / "genie_numu_2events.gtrac.root"),
                        help='Path to the GENIE gRooTracker ROOT file')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='Keep intermediate primary and particle parquet files')
    args = parser.parse_args()
    
    # Use the arguments
    simset = args.simset
    root_file_path = args.rootfile
    seed_num = args.nseed

    np.random.seed(seed_num)
    
    print(f"Using simset: {simset}")
    print(f"Using root file: {root_file_path}")
    
    # Timing for file processing and conversion in one step
    processing_start_time = time.time()
    prometheus_set, primary_set = parse_and_convert_genie(root_file_path)
    processing_end_time = time.time()
    print(f"File processing and conversion completed in {processing_end_time - processing_start_time:.2f} seconds")

    num_events = len(prometheus_set)
    primary_file_path = OUTPUT_DIR / f"primary_events_{num_events}.parquet"
    prometheus_file_path = OUTPUT_DIR / f"particle_events_{num_events}.parquet"

    primary_set, prometheus_set = inject_particles_in_cylinder(
        primary_set,  # neutrino information
        prometheus_set,  # child particle information
        cylinder_radius=100.00,  # meters
        cylinder_height=400.00,  # meters
        cylinder_center=(45.34, -58.61, -2282.0),  # meters
        detector_offset=(10.12, -8.56, -2005.37)  # Upgrade Specific Offset!
    )  # coordinates based on a cylinder in the geometric center of upgrade
    ## upgrade geometric center is: (45.34, -58.61, -2282.0)
    ## upgrade offset is (10.12, -8.56, -2005.37)
    ## The offset is something specific to a detector that prometheus calculates itself.
    #  I added a print statement that prints the detector offset. If you are not using 'icecube_upgrade_new.geo', then you need to use the detector specific offset
    primary_set, prometheus_set = rotate_particles_final(primary_set, prometheus_set)

    # Timing for serialization and saving
    save_start_time = time.time()

    # position arrays to serialized strings
    prometheus_set['position'] = prometheus_set['position'].apply(
        lambda x: [arr.tolist() for arr in x]
    )
    
    print(f"Processing {num_events} events")
    prometheus_set.to_parquet(prometheus_file_path)
    primary_set.to_parquet(primary_file_path)

    save_end_time = time.time()
    print(f"Serialization and saving completed in {save_end_time - save_start_time:.2f} seconds")

    ## Configure Prometheus for external GENIE events
    config["injection"]["name"] = "GENIE"
    config["run"]["outfile"] = str(OUTPUT_DIR / f"simulated_events_{num_events}.parquet")
    config["run"]["nevents"] = num_events
    config["injection"]["GENIE"] = config["injection"].get("GENIE", {})
    config["injection"]["GENIE"]["paths"] = config["injection"]["GENIE"].get("paths", {})
    config["injection"]["GENIE"]["inject"] = False  ## events are injected above
    config["injection"]["GENIE"]["simulation"] = {}

    ## geofile:
    config["detector"]["geo file"] = f"{RESOURCE_DIR}/geofiles/icecube_upgrade_new.geo"

    ## ppc upgrade* configuration:
    config['photon propagator']['name'] = 'PPC_UPGRADE'
    config["photon propagator"]["PPC_UPGRADE"]["paths"]["ppc_tmpdir"] = "./ppc_tmpdir" + str(simset)
    config["photon propagator"]["PPC_UPGRADE"]["paths"]["ppc_tmpfile"] = "ppc_tmp" + str(simset)
    config["photon propagator"]["PPC_UPGRADE"]["paths"]["ppctables"] = str(
        RESOURCE_DIR / "PPC_tables" / "spice_ftp-v3m"
    )


    config["photon propagator"]["PPC_UPGRADE"]["paths"]["nextgendir"] = str(
        RESOURCE_DIR / "PPC_tables" / "upgrade_tables"
    )

    config["photon propagator"]["PPC_UPGRADE"]["simulation"]["supress_output"] = True ## to suppress PPC output

    # Timing for Prometheus simulation
    sim_start_time = time.time()

    p = Prometheus(
        config,
        primary_set_parquet_path=primary_file_path,
        prometheus_set_parquet_path=prometheus_file_path
    )
    p.sim()

    if not args.keep_intermediate:
        primary_file_path.unlink(missing_ok=True)
        prometheus_file_path.unlink(missing_ok=True)

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