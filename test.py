import torch
from torch.utils.data.dataloader import DataLoader
from dataset import MyDataSet
from model import UNET
import torch.nn as nn
import matplotlib.pyplot as plt
import torch_directml

# --- CONFIG -- 
NUM_CLASSES = 3
BATCH_SIZE = 8
IMG_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#IF AMD DEVICE UNCOMMENT
DEVICE = torch_directml.device()
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




def visualize(num_samples=10):
    test_ds = MyDataSet(split='test', size=IMG_SIZE)

    model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load('unet.pth'))
    model.eval()

    mean = torch.tensor([.485, .456, .406]).view(3,1,1)
    std  = torch.tensor([.229, .224, .225]).view(3,1,1)

    _, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    axes[0, 0].set_title('Image')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 2].set_title('Prediction')

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(num_samples):
            img, mask = test_ds[i]

            pred = model(img.unsqueeze(0).to(DEVICE))
            pred_mask = pred.argmax(dim=1).squeeze(0).cpu()

            loss = loss_fn(pred, mask.unsqueeze(0).long().to(DEVICE)).item()
            confidence = pred.softmax(dim=1).max(dim=1).values.mean().item()

            display_img = (img * std + mean).permute(1, 2, 0).clamp(0, 1)

            axes[i, 0].imshow(display_img)
            axes[i, 1].imshow(mask, vmin=0, vmax=2)
            axes[i, 2].imshow(pred_mask, vmin=0, vmax=2)
            axes[i, 2].set_title(f"loss={loss:.3f}  conf={confidence:.1%}", fontsize=8)

            for ax in axes[i]:
                ax.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Saved to predictions.png")


if __name__ == "__main__":
    #test()
    visualize()












