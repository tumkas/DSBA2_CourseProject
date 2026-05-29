import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class HeartSoundDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        npy_path = row['npy_path']
        
        mel_spec = np.load(npy_path)
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        label = 1 if row['label'] == -1 else 0
        
        tensor_spec = torch.tensor(mel_spec, dtype=torch.float32)
        tensor_label = torch.tensor(label, dtype=torch.float32)
        
        return tensor_spec, tensor_label

df = pd.read_csv('processed_data.csv')

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_dataset = HeartSoundDataset(train_df)
val_dataset = HeartSoundDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)