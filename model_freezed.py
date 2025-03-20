import torch
from torchvision import models
from torch import nn
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
import torch.optim as optim
import time


device = "cuda:0" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class VGG16_piano(nn.Module):
  def __init__(self):
    super(VGG16_piano, self).__init__()
    self.vgg = models.vgg16_bn(pretrained = True)

    for param in self.vgg.features.parameters():
       param.requires_grad = False
    self.vgg.classifier[6].out_features = 2

  def forward(self, x):
    return self.vgg(x)



def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001, patience=3, lr_patience=2, factor=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=lr_patience, factor=factor, verbose=True)

    best_val_loss = float('inf')  
    patience_counter = 0

   
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() 
            outputs = model(images) 
            loss = criterion(outputs, labels) 
            loss.backward()  
            optimizer.step()  

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1) 
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - start_time
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')

        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, F1: {epoch_f1:.4f}, Time: {epoch_time:.2f}s")
        print(f"Val Loss: {val_loss}, Val accuracy: {val_acc}, Val F1: {val_f1:.4f}")
        scheduler.step(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered! 🚀")
                break
    print("Training Complete ✅")


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad(): 
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1) 
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_acc = 100 * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss / len(val_loader), val_acc, val_f1

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  
            loss = criterion(outputs, labels)  
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}% , Test F1: {f1:.4f}✅")

    return accuracy


