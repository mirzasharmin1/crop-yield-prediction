import torch
import numpy as np
from torch.utils.data import Dataset


class CropYieldDataset(Dataset):
    def __init__(self, data_df, is_train, seq_len=3, train_percentage=0.7):
        self.seq_len = seq_len
        self.is_train = is_train
        self.train_percentage = train_percentage
        self.data_cols = [col for col in data_df.columns if col not in {'Country', 'Year'}]
        self.label_cols = [col for col in data_df.columns if col.endswith('_Yield')]
        self.data_items = []
        self._populate_data(data_df)

    def _populate_data(self, data_df):
        country_groups = data_df.groupby('Country')

        for country, country_group in country_groups:
            group_sequences = []
            group_len = len(country_group)

            for start_idx in range(0, group_len - self.seq_len):
                data_rows = country_group.iloc[start_idx: start_idx + self.seq_len][self.data_cols].to_numpy()
                label_rows = country_group.iloc[start_idx + self.seq_len][self.label_cols].to_numpy()
                group_sequences.append((data_rows.astype(np.float32), label_rows.astype(np.float32)))

            train_set_len = int(len(group_sequences) * self.train_percentage)
            self.data_items.extend(
                group_sequences[:train_set_len] if self.is_train else group_sequences[train_set_len:]
            )

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        data, label = self.data_items[idx]
        return torch.tensor(data), torch.tensor(label)


class CropYieldDatasetSimple(CropYieldDataset):
    def __getitem__(self, idx):
        data, label = self.data_items[idx]
        return torch.flatten(torch.tensor(data)), torch.tensor(label)


def get_countries_and_years(data_df, seq_len, train_percentage=0.7):
    country_and_years = []
    country_groups = data_df.groupby('Country')

    for country, country_group in country_groups:
        group_len = len(country_group)
        num_sequences = group_len - seq_len
        num_valid_sequences = num_sequences - int(train_percentage * num_sequences)

        tail_rows = country_group.tail(num_valid_sequences)[['Country', 'Year']]
        country_and_years.extend(tail_rows.values.tolist())

    return country_and_years
