from src.datamodules.abstract import AbstractPLDataModule


class CodeforcesDataModule(AbstractPLDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
