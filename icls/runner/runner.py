import logging

from icls.runner.inference import RunnerInference
from icls.runner.initialize import RunnerInitialize
from omegaconf import DictConfig


class Runner(RunnerInitialize, RunnerInference):
    def __init__(self, cfg: DictConfig) -> None:

        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        self.augs = self.init_augs()
        self.dataset = self.init_dataset()
        self.dataloader = self.init_dataloader()
        self.model = self.init_model()
        self.model = self.model.to(self.cfg.device)
