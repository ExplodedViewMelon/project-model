import os
import numpy as np
import cv2

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import Accuracy
from torch import nn, optim
from torch.utils.data import Dataset

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

class FingersDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.filenames = [filename for filename in os.listdir(path)]

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.filenames[idx]
        image = cv2.imread(self.path+"/"+filename, cv2.IMREAD_GRAYSCALE)
        image = image.reshape(1,128,128) # adding singleton dimension
        image = image / 228
        label = int(filename[-6:-5])
        if filename[-5] == "R": label += 6
        
        if torch.cuda.is_available(): return (torch.from_numpy(image).type(torch.float32), torch.tensor(label).type(torch.float32))
        else: return (image, label)

class Model(LightningModule):
    def __init__(self, dropout_p, lr):
        super().__init__()

        self.dropout_p = dropout_p
        self.lr = lr

        self.backbone = nn.Sequential(
            nn.Conv2d(1,64,3), # [N, 64, 126]
            nn.LeakyReLU(),
            nn.Conv2d(64,32,3), # [N, 32, 124]
            nn.LeakyReLU(),
            nn.Conv2d(32,16,3), # [N, 16, 122]
            nn.LeakyReLU(),
            nn.Conv2d(16,8,3), # [N, 8, 120]
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 120 * 120, 128),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(128, 12)
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.type(torch.float32) # TODO: fuck this solution
        x = self.backbone(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        train_loss = self.criterium(preds, target)
        train_accuracy = self.get_accuracy(batch)
        # log metrics to wandb
        return train_loss

    def validation_step(self, batch, batch_idx): # same as training but logs it
        val_loss = self.training_step(batch, batch_idx)
        val_accuracy = self.get_accuracy(batch)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        acc = self.get_accuracy(batch)
        print(f"test accuracy: {acc}")

    def get_accuracy(self, batch):
        data, target = batch
        preds = self(data)
        accuracy = Accuracy(task = "multiclass", num_classes = 12)
        if torch.cuda.is_available(): accuracy.to("cuda")
        r = accuracy(preds, target)
        return r

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

def main():
    # define hyper parameters
    LR = 1e-4
    BATCH_SIZE = 32
    DROPOUT_P = 0.1
    print(f"hyper paramters: LR:{LR}, BATCH_SIZE:{BATCH_SIZE}, DROPOUT_P:{DROPOUT_P}")
    print("Loading data now")

    # get data
    train_set = FingersDataset("data/train/")
    val_set = FingersDataset("data/test/")
    # create dataloaders
    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=8)
    valloader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=8)
    
    print("Done loading data. Init model and fit")

    # define model and train
    model = Model(DROPOUT_P, LR) 
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min", min_delta=0.1)
    if torch.cuda.is_available(): trainer = Trainer(callbacks=[early_stopping_callback], accelerator="gpu", devices=1, max_epochs=1000)
    else: trainer = Trainer(callbacks=[early_stopping_callback], max_epochs=1000)
    trainer.fit(model, trainloader, valloader)
    print("Model succesfully trained")

if __name__ == '__main__':
    main()