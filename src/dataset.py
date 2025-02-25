import tqdm
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.vectorizer import Vectorizer
from src.directory import dataset_paths
from sklearn.model_selection import train_test_split


def merge_batches(dataloader: DataLoader):
    data_dicts = list(dataloader)
    merge_dict = {}
    for i, data_dict in enumerate(data_dicts):
        if i == 0:
            merge_dict.update(data_dict)
        else:
            for k, v in data_dict.items():
                merge_dict[k] = torch.cat([merge_dict[k], v])
    return merge_dict


class SurvivalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, group_keys=None, input_features=None, time_col='time', event_col='DEATH_EVENT'):
        self.dataset = dataset
        self.group_keys = group_keys
        self.input_features = input_features
        self.time_col = time_col
        self.event_col = event_col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_dict = {k: torch.tensor([v]) for k, v in self.dataset[idx].items()}
        return sample_dict


class SurvivalDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=128,
            num_workers=11,
            group_keys=[],
            input_features=[],
            time_col='time',
            event_col='DEATH_EVENT',
            dataset_name='faisalabad',
            **kwargs):
        super().__init__()

        self.raw_data = pd.read_csv(dataset_paths[dataset_name])
        self.n_samples = len(self.raw_data)
        self.time_col = time_col
        self.event_col = event_col
        self.dataset_name = dataset_name

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.group_keys = group_keys
        self.input_features = input_features
        self.feature_config = [*self.group_keys, *self.input_features]
        self.vectorizer = Vectorizer(feature_config=self.feature_config,
                                     dataset_name=self.dataset_name,
                                     num_bins=5)
        self.splits = dict(zip(['train', 'val', 'test'],
                               [[], [], []]))

        # temp_idx, test_idx = train_test_split(np.arange(self.n_samples), random_state=12, test_size=.2)
        # train_idx, val_idx = train_test_split(temp_idx, random_state=12, test_size=.25)
        # self.split_idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}

        train_idx, val_idx = train_test_split(list(self.raw_data.index), random_state=12, test_size=.2)
        self.split_idx = {'train': train_idx, 'val': val_idx}

    def setup(self, stage=None):
        # fit vectorize
        self.fit_vectorizer(self.raw_data.iloc[self.split_idx['train']])

        for idx, row in tqdm.tqdm(self.raw_data.iterrows(), desc=f'Processing {self.dataset_name} Dataset'):
            split = [k for k, v in self.split_idx.items() if idx in v][0]

            # add group info
            group_info = {group_key: row[group_key] for group_key in self.group_keys} if self.group_keys else {}

            # add input features
            input_features = {k: row[k] for k in self.input_features} if self.input_features else {}

            # add time & event info
            time_event_features = {k: row[k] for k in [self.time_col, self.event_col]}

            # create sample
            sample = {**time_event_features, **group_info, **input_features}

            # add it to the split
            self.splits[split].append(sample)

        dataset_kwargs = dict(group_keys=self.group_keys,
                              input_features=self.input_features)

        if stage == 'fit':
            self.train = SurvivalDataset(self.splits['train'], **dataset_kwargs)
            self.val = SurvivalDataset(self.splits['val'], **dataset_kwargs)
        if stage == 'validate':
            self.val = SurvivalDataset(self.splits['val'], **dataset_kwargs)
        if stage == 'test':
            self.test = SurvivalDataset(self.splits['test'], **dataset_kwargs)
        if stage == 'predict':
            self.predict = SurvivalDataset(self.splits['train'], **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val.__len__(), num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test.__len__(), num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.predict.__len__(), num_workers=1)

    def fit_vectorizer(self, data):
        self.vectorizer.fit(data)


class FromPandasDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_dict = {k: torch.tensor([v]) for k, v in self.dataset[idx].items()}
        return sample_dict


class XYTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=128,
            num_workers=11,
            treatment_col='T',
            outcome_col='Y',
            input_features=[],
            dataset_name='ihdp',
            raw_data=None,
            **kwargs):
        super().__init__()

        self.raw_data = pd.read_csv(dataset_paths[dataset_name]) if raw_data is None else raw_data
        self.n_samples = len(self.raw_data)
        self.dataset_name = dataset_name
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.u = None

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.input_features = input_features
        self.feature_config = [*self.input_features]
        self.splits = dict(zip(['train', 'val', 'test'],
                               [[], [], []]))

        train_val_idx, test_idx = train_test_split(list(self.raw_data.index), random_state=12, test_size=.2)
        train_idx, val_idx = train_test_split(train_val_idx, random_state=12, test_size=.2)
        self.split_idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    def setup(self, stage=None):
        self.u = self.raw_data[self.treatment_col].mean()

        # setup assumes data is standardized/one-hot encoded
        for idx, row in tqdm.tqdm(self.raw_data.iterrows(), desc=f'Processing {self.dataset_name} Dataset'):
            split = [k for k, v in self.split_idx.items() if idx in v][0]

            # add input features
            input_features = {'X': [row[k] for k in self.input_features]} if self.input_features else {}

            # add treatment and outcome info
            T_features = {'T': row[self.treatment_col]}
            Y_features = {'Y': row[self.outcome_col]}

            # create sample
            sample = {**T_features, **Y_features, **input_features}

            # add it to the split
            self.splits[split].append(sample)

        dataset_kwargs = {}

        if stage == 'fit':
            self.train = FromPandasDataset(self.splits['train'], **dataset_kwargs)
            self.val = FromPandasDataset(self.splits['val'], **dataset_kwargs)
        if stage == 'validate':
            self.val = FromPandasDataset(self.splits['val'], **dataset_kwargs)
        if stage == 'test':
            self.test = FromPandasDataset(self.splits['test'], **dataset_kwargs)
        if stage == 'predict':
            self.predict = FromPandasDataset(self.splits['test'], **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val.__len__(), num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test.__len__(), num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.predict.__len__(), num_workers=1)