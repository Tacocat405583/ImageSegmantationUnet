import torch
from torch.utils.data.dataloader import DataLoader
from dataset import MyDataSet

# --- CONFIG -- 
NUM_CLASSES = 3
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 10
IMG_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ----------




train_ds = MyDataSet()

train_loader = DataLoader(
    train_ds,
    shuffle=True,
    batch_size=BATCH_SIZE
)

len(train_loader)







