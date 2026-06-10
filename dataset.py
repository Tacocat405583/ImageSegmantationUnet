##LETS LOAD THE DATASET
import torch
from torchvision.datasets import OxfordIIITPet

class MyDataSet(torch.utils.datasets):
    def __init__(self,root='./data',split='trainval',size=256):
        self.ds = OxfordIIITPet(root=root,split=split,
                                target_types='segmentation',download=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, key):
        img,mask = self.ds[key]

        return img,mask





