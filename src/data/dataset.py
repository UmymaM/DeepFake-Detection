from torch.utils.data import Dataset
from PIL import Image
import torch


class DeepFakeDataset(Dataset):
  def __init__(self,df,transform=None):
    self.df=df
    self.transform=transform

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self,idx):
    img_path=self.df.iloc[idx,0]
    label=self.df.iloc[idx,1]
    image=Image.open(img_path).convert("RGB")
    if self.transform:
      image=self.transform(image)
    return image,torch.tensor(label, dtype=torch.float32)