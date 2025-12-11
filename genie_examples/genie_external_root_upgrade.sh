#!/bin/bash
#SBATCH --job-name=prometheus_genie
#SBATCH --output=prometheus_output_file_2006%j.out
#SBATCH --error=prometheus_err_file_2006%j.err
#SBATCH --partition=gr10_gpu  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=15G

echo "Starting the benchmark job..."


module load astro
module load hdf5/intel/1.10.4
module load intel/20.0.4


if [ -f /software/astro/anaconda/anaconda3-2020.11/etc/profile.d/conda.sh ]; then
    echo "Sourcing conda.sh..."
    source /software/astro/anaconda/anaconda3-2020.11/etc/profile.d/conda.sh
else
    echo "Could not find conda.sh!" >&2
    exit 1
fi

echo "Activating conda environment..."
conda activate myenv || { echo "Failed to activate conda environment"; exit 1; }

echo "Sourcing setup script..."
source /groups/icecube/jackp/setup_prometheus_and_genie.sh || { echo "Failed to source setup script"; exit 1; }


#cd /groups/icecube/jackp/prometheus_genie_cleaned/harvard-prometheus/examples
echo "Running benchmark script..."
unset NEXTGENDIR
#export NEXTGENDIR=../resources/PPC_tables/upgrade_tables/
#export NEXTGENDIR=../resources/PPC_tables/upgrade_internal_tables/

#python genie_external_root_upgrade2.py --simset 740123 --nseed 20123 --rootfile /groups/icecube/jackp/genie_test_outputs/output_gheps_1_to_100gev/gntp_icecube_1_to_100gev_numu_10000_seed20123.gtac.root

python genie_external_root_upgrade2.py --simset 4006 --nseed 2006 --rootfile /genie_files/genie_sim_1_to_100_GEV.gtrac.root

#python genie_external_root_upgrade2.py --simset 400 --nseed 96 --rootfile /groups/icecube/jackp/genie_test_outputs/output_gheps/gntp_icecube_numu_100.gtac.root
#gntp_icecube_numu_5.gtac.root
echo "Benchmark job completed successfully!"
