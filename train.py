import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from dataloaders import train_loader, val_loader
from cnn import HeartCNN

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)

def calculate_accuracy(y_pred_logits, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred_logits))
    correct = (y_pred_tag == y_test).sum().float()
    acc = correct / y_test.shape[0]
    return torch.round(acc * 100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HeartCNN()
model.to(device)

criterion = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

freq_masking = T.FrequencyMasking(freq_mask_param=15)
time_masking = T.TimeMasking(time_mask_param=15)

epochs = 30

for e in range(1, epochs + 1):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        X_batch = freq_masking(X_batch)
        X_batch = time_masking(X_batch)
        
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = calculate_accuracy(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    val_loss = 0
    val_acc = 0
    model.eval()
    
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            
            y_val_pred = model(X_val)
            
            v_loss = criterion(y_val_pred, y_val.unsqueeze(1))
            v_acc = calculate_accuracy(y_val_pred, y_val.unsqueeze(1))
            
            val_loss += v_loss.item()
            val_acc += v_acc.item()
            
    print(f'Epoch {e+0:03}: | Train Loss: {epoch_loss/len(train_loader):.5f} | Train Acc: {epoch_acc/len(train_loader):.3f} | Val Loss: {val_loss/len(val_loader):.5f} | Val Acc: {val_acc/len(val_loader):.3f}')

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for X_val, y_val in val_loader:
        X_val = X_val.to(device)
        y_val_pred = model(X_val)
        y_pred_tag = torch.round(torch.sigmoid(y_val_pred))
        
        all_preds.extend(y_pred_tag.cpu().numpy())
        all_targets.extend(y_val.numpy())

cm = confusion_matrix(all_targets, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print(classification_report(all_targets, all_preds, target_names=['Normal', 'Abnormal']))

torch.save(model.state_dict(), 'ultimate_model.pth')
print("Model saved to ultimate_model.pth!")