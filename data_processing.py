import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from SINDy import SINDy_readout_predictions_single_dim
from tqdm import tqdm

class EEGDatasetCached(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir
        self.cache_dir = os.path.join(args.data_dir, "processed")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.label_map = {'FEP': 0, 'TD': 1}
        self.samples = []  # list of cache file paths

        temp_storage = {}

        # --------- discover files ----------
        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".csv"):
                continue

            parts = filename.split("_")
            if parts[0] == 'EEG':
                group, sample_no, suffix = parts[1], parts[2], parts[3]
                key = f"{group}_{sample_no}"

                if key not in temp_storage:
                    temp_storage[key] = {'label': self.label_map.get(group)}

                if "EEGdata" in suffix:
                    temp_storage[key]["data"] = filename
                elif "EEGxyz" in suffix:
                    temp_storage[key]["xyz"] = filename

        # --------- process or index ----------
        for key, vals in tqdm(temp_storage.items(), desc="Processing dataset"):
            if 'data' not in vals or 'xyz' not in vals or vals['label'] is None:
                continue

            cache_file = os.path.join(self.cache_dir, f"{key}.pt")
            self.samples.append(cache_file)

            if os.path.exists(cache_file):
                continue  # already processed

            # ---- process once ----
            data_path = os.path.join(self.data_dir, vals['data'])
            xyz_path = os.path.join(self.data_dir, vals['xyz'])

            df_data = pd.read_csv(data_path, header=None)
            df_xyz = pd.read_csv(xyz_path, header=None)

            data_tensor = torch.tensor(df_data.values, dtype=torch.float32)
            xyz_tensor = torch.tensor(df_xyz.values, dtype=torch.float32)
            label = torch.tensor(vals['label'], dtype=torch.long)

            new_data_tensors = SINDy_readout_predictions_single_dim(
                data_tensor, self.args
            )

            torch.save(
                {
                    "data": new_data_tensors,
                    "xyz": xyz_tensor,
                    "label": label
                },
                cache_file
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.load(self.samples[idx], map_location="cpu")
        return sample["data"], sample["xyz"], sample["label"]
    
class RandomEEGDatasetCached(Dataset):
    def __init__(
        self,
        args,
        cache_dir,
        num_samples=100,
        num_timesteps = 130,
        num_channels = 30,
        num_classes=2,
        seed=42
    ):
        self.args = args
        self.num_timesteps = num_timesteps
        self.num_channels = num_channels
        torch.manual_seed(seed)

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples = []

        for i in tqdm(range(num_samples), desc="Creating random dataset"):
            cache_file = os.path.join(self.cache_dir, f"sample_{i}.pt")
            self.samples.append(cache_file)

            if os.path.exists(cache_file):
                continue  # already processed

            # ---- random raw data ----
            data_tensor = torch.randn(self.num_timesteps, self.num_channels, dtype=torch.float32)
            xyz_tensor = torch.randn(self.num_channels, 3)
            label = torch.randint(0, num_classes, (1,)).long()

            # ---- SAME processing as real dataset ----
            processed_data = SINDy_readout_predictions_single_dim(
                data_tensor, self.args
            )

            torch.save(
                {
                    "data": processed_data,
                    "xyz": xyz_tensor,
                    "label": label
                },
                cache_file
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.load(self.samples[idx], map_location="cpu")
        return sample["data"], sample["xyz"], sample["label"]