##LETS LOAD THE DATASET
import torch
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from PIL import Image
import numpy as np


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self,root='./data',split='trainval',size=256):
        self.ds = OxfordIIITPet(root=root,split=split,
                                target_types='segmentation',download=True)
        
        self.size = size

        self.to_tensor = transforms.ToTensor() 

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, key):
        img,mask = self.ds[key]

        img = img.resize((self.size,self.size))
        img = self.to_tensor(img)
        img = transforms.Normalize(mean=[.485,.456,.406],
                                   std=[.229,.224,.225])(img)

        mask = mask.resize((self.size,self.size),Image.Resampling.NEAREST)
        mask = np.array(mask)
        mask = torch.from_numpy(mask)

        #print(mask.size())

        mask = mask.long() - 1 #long for loss function
        

        return img,mask


if __name__ == "__main__":
    ds = MyDataSet()
    img,mask = ds[0]
    print(img.shape,img.dtype)
    print(mask.shape,mask.dtype)
    print(mask.unique())


