import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss=0.0
    all_preds=[]
    all_labels=[]
    for images,labels in tqdm(dataloader,desc="Training"):
      images=images.to(device)
      labels=labels.to(device).unsqueeze(1)
      optimizer.zero_grad()
      outputs=model(images)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
      predictions=torch.sigmoid(outputs).round().detach().cpu().numpy()
      all_preds.extend(predictions)
      all_labels.extend(labels.cpu().numpy())
    epoch_loss=running_loss/len(dataloader)
    epoch_accuracy=accuracy_score(all_labels,all_preds)
    return epoch_loss, epoch_accuracy


def validate(model,dataloader,criterion,device):
  model.eval()
  running_loss=0.0
  all_preds=[]
  all_labels=[]
  with torch.no_grad():
    for images,labels in tqdm(dataloader,desc="Validation"):
      images=images.to(device)
      labels=labels.to(device).unsqueeze(1)
      outputs=model(images)
      loss=criterion(outputs,labels)
      running_loss+=loss.item()
      predictions=torch.sigmoid(outputs).round().detach().cpu().numpy()
      all_preds.extend(predictions)
      all_labels.extend(labels.cpu().numpy())
  epoch_loss=running_loss/len(dataloader)
  epoch_accuracy=accuracy_score(all_labels,all_preds)
  return epoch_loss,epoch_accuracy