import os
import socket
import os
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

os.environ["MASTER_PORT"] = str(find_free_port())
print(f"[INFO] Using MASTER_PORT={os.environ['MASTER_PORT']}")
from os.path import join, splitext
import json
from tqdm.notebook import tqdm
import wandb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
torch.set_float32_matmul_precision('medium')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc
from torchmetrics.classification import MultilabelAccuracy

from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class CheXpertDataset(Dataset):
    def __init__(self, root_dir, patient_id_set, transform=None):
        """
        Args:
            root_dir (str): Path to the parent directory containing subdirectories (e.g., 'label_folder').
            transform (callable, optional): Optional transform to be applied on an image.
            mode (str): Either "train" or "valid" to select the correct folder.
        """
        
        # self.label_folder = os.path.join(root_dir, 'chexbert_labels')
        self.root = root_dir
        self.img_path = os.path.join(self.root, 'PNG')
        self.img_folders = [folder for folder in os.listdir(self.img_path) if splitext(folder)[1] == '']

        self.label_folder = os.path.join(self.root, 'chexbert_labels')
        self.label_path = os.path.join(self.label_folder, 'findings_fixed.json')
        self.labels = []
        self.img_paths = []
        self.img_value_exception = 'train/patient32368/study1/view1_frontal.jpg'
        self.transform = transform
        
        # load a dictionary of image paths and labels
        with open(self.label_path, 'r') as f:
            label_data = []
            for line in f:
                label_data.append(json.loads(line))

        for label_dict in label_data:
            label_list_per_sample = []
            for key, value in label_dict.items():
                if key == 'path_to_image': # save image paths
                    split_values_list = splitext(value)[0].split('/')
                    patient_id = int(split_values_list[1][7:])
                    if value != self.img_value_exception and patient_id in patient_id_set:
                        value = '/'.join(split_values_list) + '.png'
                        for folder in self.img_folders:
                            img_subfolder_path = os.path.join(os.path.join(self.img_path, folder), 'PNG')
                            img_path = os.path.join(img_subfolder_path, value)
                            if os.path.exists(img_path):
                                self.img_paths.append(img_path)
                    else:
                        break # if img_path is not saved, neither will not its label be saved
                else: # save label vectors
                    if value is None: 
                        label_list_per_sample.append(0) # if this disease is not mentioned, it is perhaps not present
                    elif value == -1:
                        label_list_per_sample.append(0) # if radiologist is uncertain, chances of having this disease or being healthy are half half
                    else:
                        label_list_per_sample.append(value) # either having this disease or not
            if len(label_list_per_sample) > 0: # empty list implies this sample is not from this set of patients
                self.labels.append(torch.tensor(label_list_per_sample, dtype=torch.long))
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")  # convert to RGB
        if self.transform:
            img = self.transform(img)
        img = img.to(torch.float32)
        label = self.labels[idx]

        return img, label

class LitDenseNetMultiLabel(pl.LightningModule):
    def __init__(self, num_classes, lr=5e-5, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        weights = DenseNet121_Weights.DEFAULT
        self.model = densenet121(weights=weights)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = MultilabelAccuracy(num_labels=num_classes, threshold=0.5)
        self.val_acc = MultilabelAccuracy(num_labels=num_classes, threshold=0.5)
        
        self.num_classes = num_classes
        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        preds = torch.sigmoid(logits)
        acc = self.train_acc(preds, y.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        preds = torch.sigmoid(logits)
        acc = self.val_acc(preds, y.int())

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets).int()

        pred_labels = (preds > 0.5).int()

        # log confusion matrix per class
        cm = multilabel_confusion_matrix(targets, pred_labels)
        for i in range(self.num_classes):
            fig, ax = plt.subplots()
            ax.imshow(cm[i], interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f"Confusion Matrix - Class {i}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            wandb.log({f"confusion_matrix_class_{i}": wandb.Image(fig)})
            plt.close(fig)

        # log ROC curves
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(targets[:, i], preds[:, i])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f"ROC Curve - Class {i}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            wandb.log({f"roc_curve_class_{i}": wandb.Image(fig)})
            plt.close(fig)

        # clear stored predictions
        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

class MultiLabelDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, num_workers=4):
        super().__init__()
        self.root = data_dir
        self.label_folder = os.path.join(self.root, 'chexbert_labels')
        self.label_path = os.path.join(self.label_folder, 'findings_fixed.json')
        self.train_patient_id_set = set()
        self.test_patient_id_set = set()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = DenseNet121_Weights.DEFAULT.transforms()

    def setup(self, stage=None):
        # load a dictionary of image paths and labels
        with open(self.label_path, 'r') as f:
            label_data = []
            for line in f:
                label_data.append(json.loads(line))

        for label_dict in label_data:
            split_values_list = splitext(label_dict['path_to_image'])[0].split('/')
            mode = split_values_list[0] # train or test
            patient_id = int(split_values_list[1][7:])
            if mode == 'train':
                self.train_patient_id_set.add(patient_id)
            elif mode == 'test':
                self.test_patient_id_set.add(patient_id)
        
        train_patient_id_list = list(self.train_patient_id_set)
        trainset_idx = np.random.choice(np.arange(len(train_patient_id_list)), int(0.75*len(train_patient_id_list)), replace=False)
        train_patient_id_set = set([train_patient_id_list[idx] for idx in trainset_idx])
        val_patient_id_set = self.train_patient_id_set - train_patient_id_set

        self.train_set = CheXpertDataset(root_dir=self.root, patient_id_set=train_patient_id_set, transform=self.transform)
        self.val_set = CheXpertDataset(root_dir=self.root, patient_id_set=val_patient_id_set, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

def main():
    data_folder = 'CheXpert Plus'

    model = LitDenseNetMultiLabel(num_classes=14)
    data = MultiLabelDataModule(data_dir=data_folder)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="lr_5e-5_batch_size_16_best-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    wandb_logger = WandbLogger(project="chexpert_multilabel", name="lr_5e-5_batch_size_16")

    trainer = pl.Trainer(
        max_epochs=8,
        accelerator='gpu',
        devices='auto',  # or specify a list like devices=[0,1]
        # strategy='ddp',  # distributed data parallel
        precision="16-mixed",     # optional mixed precision
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, data)

if __name__ == '__main__':
    main()