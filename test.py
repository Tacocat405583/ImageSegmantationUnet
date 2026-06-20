import torch
from torch.utils.data.dataloader import DataLoader
from dataset import MyDataSet
from model import UNET
import torch.nn as nn

# --- CONFIG -- 
NUM_CLASSES = 3
BATCH_SIZE = 8
IMG_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ----------

def test():
    test_ds = MyDataSet(split='test',size=IMG_SIZE)

    test_data_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE
    )

    model = UNET(in_channels=3,out_channels=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load('unet.pth'))

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        test_loss = 0

        for batch, (X,Y) in enumerate(test_data_loader):
                X,Y = X.to(DEVICE),Y.to(DEVICE)

                y_pred = model(X)

                loss = loss_fn(y_pred,Y)
                test_loss += loss


                if batch % 400 == 0:
                    print(f"Looked at {batch*len(X)}/{len(test_data_loader.dataset)} samples")

        test_loss /= len(test_data_loader)
        print(f"Test loss: {test_loss:.4f}")




if __name__ == "__main__":
    test()












