#!/bin/bash
#SBATCH --job-name=ACCESS_to_ERA5
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=ACCESS_to_ERA5.out
#SBATCH --error=ACCESS_to_ERA5.out
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --cpus-per-task=24
#SBATCH -A bqu@h100
#SBATCH -C h100

module purge
module load pytorch-gpu/py3/2.7.0
module load openmpi/4.1.6-cuda

export PYTHONPATH=/lustre/fswork/projects/rech/bqu/uyv87oi/libs:$PYTHONPATH

# ---------------------------
# 1. Convert NetCDF to tensors
# ---------------------------
srun python -u nc_to_tensors.py \
    --path_GCM data/sfcWind_FR_ACCESS-ESM1-5_1980_2022.nc \
    --path_REA data/ERA5_1980_2022.nc \
    --variables_GCM GCM \
    --variables_REA ERA5 \
    --interpolation spectral \
    --max_train_data 2002 \
    --name_GCM ACCESS \
    --name_REA ERA5

# ---------------------------
# 2. Run classifier to get optimal r_cut
# ---------------------------
# Capture r_cut from stdout or write to file in classifier script
R_CUT=$(srun python -u classifier_serpentflow.py \
    --path_train_A data/ACCESS_train.pt \
    --path_train_B data/ERA5_train.pt \
    --path_test_A data/ACCESS_test.pt \
    --path_test_B data/ERA5_test.pt \
    --get_r_cut)
echo "Detected r_cut = $R_CUT"

# ---------------------------
# 3. Train flow matching model
# ---------------------------
srun python -u train_serpentflow.py \
    --path_B data/ERA5_train.pt \
    --r_cut $R_CUT \
    --name_config ERA5 \
    --save_name ACCESS_to_ERA5 \
    --num_generations 4

# ---------------------------
# 4. Run inference
# ---------------------------
srun python -u inference_serpentflow.py \
    --path_A data/ACCESS_test.pt \
    --r_cut $R_CUT \
    --name_config ERA5 \
    --save_name_model ACCESS_to_ERA5 \
    --save_name ACCESS_to_ERA5_output
