from importlib import import_module

import icls.types as T
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


class Builder:
    def build_blocks(self, object_cfg: T.ListConfig) -> None:

        """
        Build blocks composing the object.
        Args:
            object_cfg (T.ListConfig): The list of blocks
        """

        for block_cfg in object_cfg:

            var_name, cls_fullname = block_cfg.popitem()
            _, block_cfg = block_cfg.popitem()

            module_path, cls_name = cls_fullname.split(" - ")
            module = import_module(module_path)
            cls = getattr(module, cls_name)

            if type(block_cfg) == DictConfig:
                setattr(self, var_name, cls(**block_cfg))

            elif type(block_cfg) == ListConfig:
                setattr(self, var_name, cls(block_cfg))
