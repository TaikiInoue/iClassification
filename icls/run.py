import os

import hydra
from omegaconf import DictConfig

from icls.runner import Runner


@hydra.main(config_path="yamls", config_name="imagenet")
def main(cfg: DictConfig) -> None:

    os.rename(".hydra", "hydra")

    runner = Runner(cfg)
    runner.inference()


if __name__ == "__main__":
    main()
