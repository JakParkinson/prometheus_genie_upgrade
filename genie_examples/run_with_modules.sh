#!/bin/bash
module load astro
module load hdf5/intel/1.10.4
module load intel/20.0.4
source /software/astro/anaconda/anaconda3-2020.11/etc/profile.d/conda.sh
conda activate myenv
source /groups/icecube/jackp/setup_prometheus_and_genie.sh

# Launch jupyter with this environment
exec "$@"
