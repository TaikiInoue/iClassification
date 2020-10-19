from importlib import import_module

import icls.albu as albu
from icls.albu import Compose
from omegaconf import DictConfig
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


class RunnerInitialize:

    cfg: DictConfig
    dataset: Dataset
    augs: Compose

    def init_augs(self) -> Compose:

        augs = albu.load(self.cfg.augs.yaml, data_format="yaml")
        albu.save(augs, "hydra/augs.yaml", data_format="yaml")
        return augs

    def init_dataloader(self) -> DataLoader:

        attr = self._get_attr(self.cfg.dataloader.name)
        return attr(**self.cfg.dataloader.args, dataset=self.dataset)

    def init_dataset(self) -> Dataset:

        attr = self._get_attr(self.cfg.dataset.name)
        return attr(**self.cfg.dataset.args, augs=self.augs)

    def init_model(self) -> Module:

        attr = self._get_attr(self.cfg.model.name)
        return attr(**self.cfg.model.args)

    def _get_attr(self, fullname: str):

        module_path, attr_name = fullname.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)
