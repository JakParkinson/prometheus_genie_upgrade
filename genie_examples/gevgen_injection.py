import os

import logging
import subprocess
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        "-e", "5,40", ## in GeV
        "-f", "x^-2", ## Energy flux
        "--seed", "16",
        "--cross-sections", "/groups/icecube/jackp/genie-3.4.2/ice_numu_cross_sections.xml", ## combined H and O16 cross section filex into 1
       # "--event-generator-list", "Default",
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