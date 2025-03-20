import torch
from torchvision import models
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_freezed import VGG16_piano
from model_freezed import train_model
from model_freezed import test_model


device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=f"vase_teapot/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(root=f"vase_teapot/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

val_dataset = datasets.ImageFolder(root=f"vase_teapot/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

PATH = f"models/teapot_vase_f1score_maro_vgg16.pth"


model = VGG16_piano().to(device)


train_model(model, train_loader, learning_rate=0.0001, val_loader=val_loader)
torch.save(model.state_dict(), PATH)
test_model(model, test_loader)
