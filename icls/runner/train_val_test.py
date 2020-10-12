import icls.types as T
import torch
from tqdm import tqdm


class RunnerTrainValTest:

    cfg: T.DictConfig
    criterion: T.Loss
    dataloader_dict: T.Dict[str, T.DataLoader]
    log: T.Logger
    model: T.Module
    optimizer: T.Optimizer

    def train(self):

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

    def val(self):
        pass

    def test(self):

        self.model.eval()
        pbar = tqdm(self.dataloader_dict["test"], desc="test")
        for i, data_dict in enumerate(pbar):

            img = data_dict["image"].to(self.cfg.device)
            pred = self.model(img)
            pred = pred.argmax(dim=1)
