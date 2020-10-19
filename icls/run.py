import os
import sys

import hydra
from omegaconf import DictConfig

from icls.runner import Runner

config_path = sys.argv[1]
sys.argv.pop(1)


@hydra.main(config_path)
def main(cfg: DictConfig) -> None:

    os.rename(".hydra", "hydra")

    runner = Runner(cfg)
    runner.run_train()
    runner.run_test()


if __name__ == "__main__":
    main()
