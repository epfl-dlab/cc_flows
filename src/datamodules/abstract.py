from abc import ABC
from typing import Optional, Callable

import hydra
import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class AbstractPLDataModule(LightningDataModule, ABC):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    def __init__(self, seed: int, collate_fn: Callable = None, num_workers: int = 0, **kwargs):
        """

        Parameters
        ----------
        seed : Random seed
        collate_fn : The collate function which is model specific
        num_workers : Setting num_workers as a positive integer will turn on multi-process data loading with the specified number of loader worker processes

        kwargs: dataset specific parameters

        Returns
        -------
        An instance of the Grid dataset that extends pytorch_lightning.DataModule
        """
        super().__init__()
        self.collate_fn = collate_fn

        # Concerning the loaders
        self.num_workers = num_workers
        self.seed = seed

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.dataset_parameters = kwargs["dataset_parameters"]

    def set_collate_fn(self, collate_fn):
        self.collate_fn = collate_fn

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test.
        Note that the result of hydra instantiation inherits abstract_grid_dataset for which we have:
            def __getitem__(self, idx):
                return {"id": self.data[idx][0], "text": self.data[idx][1]}

        So when a sample from such dataset is picked (self.data_train/val/test), the sample has the
        above form, and to extract its text, we need to pass the keyword "text".
        """
        assert stage in set(["fit", "validate", "test", None])

        if stage == "fit" or stage is None:
            self.data_train = hydra.utils.instantiate(self.dataset_parameters["train"]["dataset"])
            self.data_val = hydra.utils.instantiate(self.dataset_parameters["val"]["dataset"])

        if (stage == "validate" or stage is None) and self.data_val is None:
            self.data_val = hydra.utils.instantiate(self.dataset_parameters["val"]["dataset"])

        if stage == "test" or stage is None:
            self.data_test = hydra.utils.instantiate(self.dataset_parameters["test"]["dataset"])

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.dataset_parameters["train"]["dataloader"]["batch_size"],
            num_workers=self.dataset_parameters["train"]["dataloader"]["num_workers"],
            collate_fn=self.collate_fn,
            drop_last=False,
            shuffle=True,
            generator=g,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.dataset_parameters["val"]["dataloader"]["batch_size"],
            num_workers=self.dataset_parameters["val"]["dataloader"]["num_workers"],
            collate_fn=self.collate_fn,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.dataset_parameters["test"]["dataloader"]["batch_size"],
            num_workers=self.dataset_parameters["test"]["dataloader"]["num_workers"],
            collate_fn=self.collate_fn,
            drop_last=False,
            shuffle=False,
        )
