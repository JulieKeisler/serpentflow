"""
Convert GCM and Reanalysis NetCDF data to Torch tensors for SerpentFlow training.

Example usage:
--------------
python nc_to_tensors.py \
    --path_GCM data/GCM.nc \
    --path_REA data/REA.nc \
    --variables_GCM temp \
    --variables_REA temp \
    --interpolation spectral \
    --max_train_data 2000 \
    --name_GCM GCM \
    --name_REA REA
"""

import argparse
import xarray as xr
import torch
import os
from utils.data_utils import upsample_and_lowpass

def main():
    parser = argparse.ArgumentParser(description="Convert .nc GCM and REA data to torch tensors")
    parser.add_argument("--path_GCM", type=str, required=True)
    parser.add_argument("--path_REA", type=str, required=True)
    parser.add_argument("--variables_GCM", type=str, required=True)
    parser.add_argument("--variables_REA", type=str, required=True)
    parser.add_argument("--interpolation", type=str, default="spectral", choices=["linear", "spectral"])
    parser.add_argument("--max_train_data", type=int, default=2000)
    parser.add_argument("--name_GCM", type=str, required=True)
    parser.add_argument("--name_REA", type=str, required=True)

    args = parser.parse_args()

    print("Loading datasets...")
    coarse_gcm = xr.load_dataset(args.path_GCM)[args.variables_GCM].load()
    superres_reanalysis = xr.load_dataset(args.path_REA)[args.variables_REA].load()
    print(f"Loaded GCM shape: {coarse_gcm.shape}, REA shape: {superres_reanalysis.shape}")

    # Interpolation / upsampling
    if args.interpolation == "linear":
        print("Applying linear interpolation...")
        superres_gcm = coarse_gcm.interp(lat=superres_reanalysis.lat, lon=superres_reanalysis.lon)
    elif args.interpolation == "spectral":
        print("Applying spectral upsampling + low-pass filter...")
        superres_gcm_tensor = upsample_and_lowpass(
            torch.tensor(coarse_gcm.data, dtype=torch.float32),
            target_size=(len(superres_reanalysis.lat), len(superres_reanalysis.lon)),
            cutoff=0.06
        )
        superres_gcm = xr.Dataset(
            data_vars={"GCM": (["time", "lat", "lon"], superres_gcm_tensor)},
            coords={
                "time": coarse_gcm.time.data,
                "lat": superres_reanalysis.lat.data,
                "lon": superres_reanalysis.lon.data
            }
        )
    print(f"Upsampled GCM shape: {superres_gcm['GCM'].shape}")

    # Split train/test
    print("Splitting datasets into train/test...")
    superres_gcm_train = superres_gcm.sel(time=superres_gcm.time.dt.year < args.max_train_data)
    superres_gcm_test = superres_gcm.sel(time=superres_gcm.time.dt.year >= args.max_train_data)
    superres_rea_train = superres_reanalysis.sel(time=superres_reanalysis.time.dt.year < args.max_train_data)
    superres_rea_test = superres_reanalysis.sel(time=superres_reanalysis.time.dt.year >= args.max_train_data)

    # Save tensors
    print("Saving tensors to data/ folder...")

    os.makedirs("data", exist_ok=True)

    # Mapping tensors to filenames
    tensors_to_save = {
        f"{args.name_GCM}_train.pt": superres_gcm_train.transpose('time', 'lat', 'lon').to_array(),
        f"{args.name_GCM}_test.pt": superres_gcm_test.transpose('time', 'lat', 'lon').to_array(),
        f"{args.name_REA}_train.pt": superres_rea_train.transpose('time', 'lat', 'lon'),
        f"{args.name_REA}_test.pt": superres_rea_test.transpose('time', 'lat', 'lon')
    }

    for filename, ds in tensors_to_save.items():
        # Transpose to (time, lat, lon) and convert to torch tensor
        tensor = torch.tensor(ds.data, dtype=torch.float32).squeeze()
        
        # If single variable, add channel dimension
        tensor = tensor.unsqueeze(1)
        
        # Save tensor
        path = os.path.join("data", filename)
        torch.save(tensor, path)
        print(f"Saved {path}, tensor shape: {tensor.shape}")

    print("Done! All tensors saved successfully.")

if __name__ == "__main__":
    main()
