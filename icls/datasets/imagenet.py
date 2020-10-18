from pathlib import Path

import cv2
import icls.types as T
from torch.utils.data import Dataset


class ImagenetDataset(Dataset):
    def __init__(self, prefix: str, augs: T.Compose) -> None:

        self.prefix = Path(prefix)
        self.augs = augs

        with open("data/imagenet/meta/val.txt") as f:
            lines = f.readlines()

        self.filename_list = []
        self.label_list = []
        for line in lines:
            filename, label = line.strip().split(" ")
            self.filename_list.append(filename)
            self.label_list.append(label)

    def __getitem__(self, idx: int) -> dict:

        filename = self.filename_list[idx]
        label = self.label_list[idx]

        img = cv2.imread(self.prefix / filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data_dict = self.augs(image=img)
        data_dict["label"] = label

        return data_dict

    def __len__(self) -> int:

        return len(self.filename_list)
