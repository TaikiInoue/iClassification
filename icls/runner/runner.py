import logging

from icls.runner.initialize import RunnerInitialize
from icls.runner.train_val_test import RunnerTrainValTest
from omegaconf import DictConfig


class Runner(
    RunnerInitialize,
    RunnerTrainValTest,
):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        self.augs_dict = {}
        self.dataset_dict = {}
        self.dataloader_dict = {}
        for data_type in ["val"]:
            self.augs_dict[data_type] = self.init_augs(data_type)
            self.dataset_dict[data_type] = self.init_dataset(data_type)
            self.dataloader_dict[data_type] = self.init_dataloader(data_type)

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.criterion = self.init_criterion()
