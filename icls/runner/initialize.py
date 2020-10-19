from importlib import import_module

import icls.albu as albu
from icls.albu import Compose
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset


class RunnerInitialize:

    cfg: DictConfig

    def init_augs(self, data_type: str) -> Compose:

        augs = albu.load(self.cfg.augs[data_type].yaml, data_format="yaml")

        for update in self.cfg.augs[data_type].updates:
            for i, aug in enumerate(augs):
                if aug.__class__.__name__ == update.name:
                    for k, v in update.args.items():
                        setattr(augs[i], k, v)

        albu.save(augs, f"hydra/{data_type}_augs.yaml", data_format="yaml")

        return augs

    def init_criterion(self) -> Module:

        return self._init(self.cfg.criterion)

    def init_dataloader(self, data_type: str) -> DataLoader:

        return self._init(self.cfg.dataloder[data_type])

    def init_dataset(self, data_type: str) -> Dataset:

        return self._init(self.cfg.dataset[data_type])

    def init_model(self) -> Module:

        return self._init(self.cfg.model)

    def init_optimizer(self) -> Optimizer:

        return self._init(self.cfg.optimizer)

    def init_scheduler(self):

        return self._init(self.cfg.scheduler)

    def _init(self, cfg: DictConfig):

        fullname = cfg.name
        module_path, attr_name = fullname.split(" - ")
        module = import_module(module_path)
        attr = getattr(module, attr_name)

        if cfg.args:
            return attr(**cfg.args)
        else:
            return attr()
