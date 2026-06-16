import torch
from torch.utils.data.dataloader import DataLoader
from dataset import MyDataSet
from model import UNET
import torch.nn as nn
from tqdm.auto import tqdm

# --- CONFIG -- 
NUM_CLASSES = 3
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 4
IMG_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ----------




def train():
    train_ds = MyDataSet()

    train_data_loader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    model = UNET(in_channels=3,out_channels=NUM_CLASSES).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    for epoch in tqdm(range(EPOCHS)):
        print(f"Epoch number: {epoch}")

        train_loss = 0

        model.train()

        for batch,(X,y) in enumerate(train_data_loader):
            X,y = X.to(DEVICE),y.to(DEVICE)

            y_pred = model(X)

            loss = loss_fn(y_pred,y)
            train_loss += loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if batch % 400 == 0:
                print(f"Looked at {batch*len(X)}/{len(train_data_loader.dataset)} samples")

        train_loss /= len(train_data_loader)
        print(f"Train loss: {train_loss:.4f}")

    torch.save(model.state_dict(), "unet.pth")



if __name__ == "__main__":
    train()

    








