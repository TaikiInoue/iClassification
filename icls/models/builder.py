from abc import ABC
from importlib import import_module

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


class Builder(ABC):
    def build_blocks_from_cfg(self, cfg: ListConfig) -> None:

        """
        Build blocks composing the object.
        Args:
            cfg (ListConfig): The list of block configs
        """

        for block_cfg in cfg:

            var_name, cls_fullname = block_cfg.popitem()
            _, block_cfg = block_cfg.popitem()

            module_path, cls_name = cls_fullname.split(" - ")
            module = import_module(module_path)
            cls = getattr(module, cls_name)

            if type(block_cfg) == DictConfig:
                setattr(self, var_name, cls(**block_cfg))

            elif type(block_cfg) == ListConfig:
                setattr(self, var_name, cls(block_cfg))
