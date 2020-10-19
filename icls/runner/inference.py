from logging import Logger
from typing import Dict

import torch
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class RunnerInference:

    cfg: DictConfig
    dataloader: Dict[str, DataLoader]
    model: Module

    def inference(self) -> None:

        self.model.eval()
        pbar = tqdm(self.dataloader, desc="inference")
        for i, data_dict in enumerate(pbar):

            img = data_dict["image"].to(self.cfg.device)
            label = data_dict["label"]
            _, pred = self.model(img)
            pred = pred.argmax(dim=1)  # pred.shape -> (batch_size, num_classes)
