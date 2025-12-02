# SerpentFlow

**SerpentFlow** (SharEd-structuRe decomPosition for gEnerative domaiN adapTation) is a framework for **unpaired domain alignment**.  
It separates shared low-frequency structures from domain-specific high-frequency content and uses **Flow Matching** for generative modeling.

---

## 1. Installation & Dependencies

Install required Python packages:

```bash
pip install torch torchvision torchmetrics torchdiffeq xarray numpy
```
Clone the repository:

```bash
git clone <repo_url>
cd serpentflow
```

## 2. Preprocessing: NetCDF to Torch tensors

Use nc_to_tensors.py to convert GCM and reanalysis NetCDF files to .pt tensors:

```bash
python nc_to_tensors.py \
    --path_GCM data/gcm.nc \
    --path_REA data/rea.nc \
    --variables_GCM var1 var2 \
    --variables_REA var1 var2 \
    --interpolation spectral \
    --max_train_data 2000 \
    --name_GCM GCM \
    --name_REA REA
```

Outputs:

```
data/GCM_train.pt
data/GCM_test.pt
data/REA_train.pt
data/REA_test.pt
````

- ```interpolation```: "spectral" (FFT-based upsampling + low-pass) or "linear"

- ```max_train_data```: split year threshold for train/test

## 3. Selecting Optimal Cutoff Frequency (r_cut) with a classifier

SerpentFlow includes a binary classifier to identify the optimal r_cut separating low- and high-frequency content.

### Steps:

- #### Prepare datasets:
    - Class 0: low-pass filtered images
    - Class 1: original images

- #### Train classifier

- #### Select r_cut:

    - Apply candidate low-pass filters to your dataset
    - Evaluate with classifier
    - Choose r_cut where classifier accuracy drops below a threshold (e.g., 60%)

## 4. Training Flow Matching (SerpentFlow)

Once r_cut is selected:
```bash
python train_serpentflow.py \
    --path_B data/REA_train.pt \
    --r_cut 5 \
    --name_config ERA5 \
    --save_name experiment \
    --num_generations 4
```

- ```path_B```: target domain dataset

- ```r_cut```: low-pass cutoff frequency

- ```name_config```: dataset configuration (ERA5, MNIST, CIFAR10, **add your own** to ```utils/config.py```)

- ```save_name```: checkpoint prefix

- ```num_generations```: number of samples to generate after training

## 5. Inference / Pseudo-pair Generation

Generate outputs for a source dataset:

```bash
python inference_serpentflow.py \
    --path_A data/GCM_test.pt \
    --r_cut 5 \
    --name_config ERA5 \
    --save_name_model experiment \
    --save_name GCM2REA_output
```

Outputs will be saved as:

```bash
data/results/GCM2REA_output.pt
data/results/GCM2REA_output_grid.png
```

- ```path_A```: source dataset

- ```r_cut```: same cutoff used during training

- ```save_name_model```: checkpoint prefix from training

- ```save_name```: output prefix

## 6. Directory Structure

```
SerpentFlow/
├─ data/                       # Preprocessed tensors and outputs
├─ checkpoints/                # Saved model checkpoints
├─ src/
│  ├─ datasets.py              # Dataset classes
│  ├─ train.py                 # Flow matching training loop
│  ├─ inference.py             # Inference pipeline
├─ utils/
│  ├─ data_utils.py            # Low/high frequency decomposition
│  ├─ models.py                # UNet, classifier
│  ├─ training_utils.py        # EMA, training helpers
│  ├─ inference_utils.py       # ODE solver, grid generation
│  ├─ plots.py                 # Image grid saving
│  ├─ config.py                # Model/training configs
├─ nc_to_tensors.py            # NetCDF → Torch tensors
├─ classifier_serpentflow.py   # Binary classifier for r_cut selection
├─ train_serpentflow.py        # Main training script
├─ inference_serpentflow.py    # Main inference script
````

## 7. Notes

- All tensors are assumed to be in (N, C, H, W) format, with C=1. /!\/!\/!\ Multivariate downscaling has not been tried yet.

- EMA is used for stabilizing flow matching

- Classifier is used only for choosing r_cut, not part of generative training

- Low-pass/high-pass decomposition is FFT-based (we might explore other options)

## Citation

If you use this code in your research, please cite:

Julie Keisler, Anastase Charantonis, Yannig Goude, Boutheina Oueslati, Claire Monteleoni.  
**Generative Unpaired Domain Alignment via Shared-Structure Decomposition**. Preprint, 2025.
