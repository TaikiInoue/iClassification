from logging import Logger
from typing import Dict

import torch
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class RunnerTrainValTest:

    cfg: DictConfig
    criterion: Module
    dataloader_dict: Dict[str, DataLoader]
    log: Logger
    model: Module
    optimizer: Optimizer

    def train(self) -> None:

        self.model.train()
        pbar = tqdm(range(1, self.cfg.run.train.epochs + 1), desc="train")
        for epoch in pbar:

            self.log.info(f"epoch - {epoch}")
            cumulative_loss = 0
            for data_dict in self.dataloader_dict["train"]:

                self.optimizer.zero_grad()
                img = data_dict["image"].to(self.cfg.device)
                label = data_dict["label"].long().to(self.cfg.device)

                pred = self.model(img)
                loss = self.criterion(pred, label)
                loss.backward()
                cumulative_loss += loss.item()

            epoch_loss = cumulative_loss / len(self.dataloader_dict["train"])
            self.log.info(f"loss - {epoch_loss}")
            self.scheduler.step()

            if epoch % 100 == 0:
                self.run_test()
                self.model.train()

        torch.save(self.model.state_dict(), f"{self.cfg.model.name}.pth")

    def val(self) -> None:

        self.model.eval()
        pbar = tqdm(self.dataloader_dict["val"], desc="val")
        for i, data_dict in enumerate(pbar):

            img = data_dict["image"].to(self.cfg.device)
            label = data_dict["label"]
            pred = self.model(img)
            pred = pred.argmax(dim)

    def test(self) -> None:

        self.model.eval()
        pbar = tqdm(self.dataloader_dict["test"], desc="test")
        for i, data_dict in enumerate(pbar):

            img = data_dict["image"].to(self.cfg.device)
            pred = self.model(img)
            pred = pred.argmax(dim=1)
