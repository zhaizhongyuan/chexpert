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
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights, resnet152,ResNet152_Weights
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class CheXpertClsDataset(Dataset):
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
                    patient_id = split_values_list[1][7:]
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

class CheXpertGenDataset(Dataset):
    def __init__(self, root_dir, patient_id_set, transform, model, target_layers):
        """
        Args:
            root_dir (str): Path to the parent directory containing subdirectories (e.g., 'label_folder').
            transform (callable, optional): Optional transform to be applied on an image.
            mode (str): Either "train" or "valid" to select the correct folder.
        """
        self.root = root_dir
        self.imgs_path = os.path.join(self.root, 'PNG')
        self.img_folders = [folder for folder in os.listdir(self.imgs_path) if splitext(folder)[1] == '']

        self.label_path = os.path.join(self.root, 'df_chexpert_plus_240401.csv')
        # load a dataframe of image paths and radiologist report texts
        self.chexpert_df = pd.read_csv(self.label_path)

        self.patient_id_study_num_set = set()
        self.patient_id_study_num_list = []
        self.img_dict = {}
        self.text_dict = {}
        self.img_value_exception = 'train/patient32368/study1/view1_frontal.jpg'
        self.transform = transform

        self.model = model
        self.target_layers = target_layers

        for idx in tqdm(range(len(self.chexpert_df))):
            img_value = self.chexpert_df.iloc[idx]['path_to_image']
            if img_value != self.img_value_exception:
                img_value_list = splitext(img_value)[0].split('/')
                for folder in self.img_folders:
                    img_subfolder_path = os.path.join(os.path.join(self.imgs_path, folder), 'PNG')
                    img_path_temp = os.path.join(img_subfolder_path, '/'.join(img_value_list) + '.png')
                    if os.path.exists(img_path_temp):
                        img_path = img_path_temp

                patient_id = img_value_list[1][7:]
                study_num = img_value_list[2]
                if patient_id in patient_id_set:
                    patient_id_study_num = f'patient{patient_id}/{study_num}'
                    self.patient_id_study_num_set.add(patient_id_study_num)

                    if patient_id_study_num not in self.img_dict:
                        # add list of images to the same study of a patient
                        img_paths_list = [img_path]
                        self.img_dict[patient_id_study_num] = img_paths_list

                        # add findings and impression texts to a list
                        section_reports_dict = {}
                        section_findings = self.chexpert_df.iloc[idx]['section_findings']
                        if isinstance(section_findings, str) and len(section_findings.split()) >= 2:
                            section_findings = section_findings.strip().replace('\n', '').replace('..', '.') # strips mostly '\n'
                            section_findings = '. '.join([s.strip().capitalize() for s in section_findings.split('.')]).strip() # lower caps
                            section_reports_dict['findings'] = section_findings
                        else:
                            section_reports_dict['findings'] = ''
                        section_impression = self.chexpert_df.iloc[idx]['section_impression']
                        if isinstance(section_impression, str) and len(section_impression.split()) >= 2:
                            section_impression = section_impression.strip().replace('\n', '').replace('..', '.') # strips mostly '\n'
                            section_impression = '. '.join([s.strip().capitalize() for s in section_impression.split('.')]).strip() # lower caps
                            section_reports_dict['impression'] = section_impression
                        else:
                            section_reports_dict['impression'] = ''
                        self.text_dict[patient_id_study_num] = section_reports_dict
                    else:
                        self.img_dict[patient_id_study_num].append(img_path)
        self.patient_id_study_num_list = list(self.patient_id_study_num_set)

    def __len__(self):
        assert len(self.img_dict) == len(self.text_dict)
        assert len(self.patient_id_study_num_list) == len(self.img_dict)
        return len(self.img_dict)

    def __getitem__(self, idx):
        item = {}
        # open images and transform to tensor
        patient_id_study_num = self.patient_id_study_num_list[idx]
        img_paths = self.img_dict[patient_id_study_num]
        cam_aug_img_list = []
        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")  # convert to RGB
            img = self.transform(img)
            img = img.to(torch.float32)
            # cam_aug_list = [img]
            # for class_idx in range(14):
            #     targets = [ClassifierOutputTarget(class_idx)]
            #     with GradCAM(model=self.model, target_layers=self.target_layers) as cam:
            #         cam_aug_list.append(cam(input_tensor=torch.unsqueeze(img, dim=0), targets=targets))
            # cam_aug_img = torch.cat(cam_aug_list, dim=0)
            # cam_aug_img_list.append(cam_aug_img)
            cam_aug_img_list.append(img)
        item['images'] = cam_aug_img_list # list of img tensors
        item['findings'] = self.text_dict[patient_id_study_num]['findings']
        item['impression'] = self.text_dict[patient_id_study_num]['impression']
        return item

    def collate_fn(self, batch):
        """
        batch: list of items returned by __getitem__
        """
        # turn list-of-dicts into dict-of-lists
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
        # stack per-sample
        images_stacked = [
            torch.stack(img_list, dim=0)      # => [n_views_i, C, H, W]
            for img_list in batch_dict['images']
        ]
        # pad across the batch dimension (n_views) to the max n_views in this batch
        padded = torch.nn.utils.rnn.pad_sequence(
            images_stacked,
            batch_first=True,    # => [B, max_n_views, C, H, W]
            padding_value=0.0
        )
        batch_dict['images'] = padded
        return batch_dict

class CheXpertClsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform=None, batch_size=16, num_workers=4):
        super().__init__()
        self.root = data_dir
        self.label_folder = os.path.join(self.root, 'chexbert_labels')
        self.label_path = os.path.join(self.label_folder, 'findings_fixed.json')
        self.train_patient_id_set = set()
        self.test_patient_id_set = set()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        # load a dictionary of image paths and labels
        with open(self.label_path, 'r') as f:
            label_data = []
            for line in f:
                label_data.append(json.loads(line))

        for label_dict in label_data:
            split_values_list = splitext(label_dict['path_to_image'])[0].split('/')
            mode = split_values_list[0] # train or test
            patient_id = split_values_list[1][7:]
            if mode == 'train':
                self.train_patient_id_set.add(patient_id)
            elif mode == 'valid':
                self.test_patient_id_set.add(patient_id)
        
        if stage == 'fit' or stage is None:
            train_patient_id_list = list(self.train_patient_id_set)
            trainset_idx = np.random.choice(np.arange(len(train_patient_id_list)), int(0.75*len(train_patient_id_list)), replace=False)
            train_patient_id_set = set([train_patient_id_list[idx] for idx in trainset_idx])
            val_patient_id_set = self.train_patient_id_set - train_patient_id_set

            self.train_set = CheXpertClsDataset(root_dir=self.root, patient_id_set=train_patient_id_set, transform=self.transform)
            self.val_set = CheXpertClsDataset(root_dir=self.root, patient_id_set=val_patient_id_set, transform=self.transform)

        if stage == 'test' or stage is None:
            self.test_set = CheXpertClsDataset(root_dir=self.root, patient_id_set=self.test_patient_id_set, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

class CheXpertGenDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, model, target_layers, transform=None, batch_size=16, num_workers=4):
        super().__init__()
        self.root = data_dir
        self.label_folder = os.path.join(self.root, 'chexbert_labels')
        self.label_path = os.path.join(self.label_folder, 'findings_fixed.json')
        self.train_patient_id_set = set()
        self.test_patient_id_set = set()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self.model = model
        self.target_layers = target_layers

    def setup(self, stage=None):
        # load a dictionary of image paths and labels
        with open(self.label_path, 'r') as f:
            label_data = []
            for line in f:
                label_data.append(json.loads(line))

        for label_dict in label_data:
            split_values_list = splitext(label_dict['path_to_image'])[0].split('/')
            mode = split_values_list[0] # train or test
            patient_id = split_values_list[1][7:]
            if mode == 'train':
                self.train_patient_id_set.add(patient_id)
            elif mode == 'valid':
                self.test_patient_id_set.add(patient_id)
        
        if stage == 'fit' or stage is None:
            train_patient_id_list = list(self.train_patient_id_set)
            trainset_idx = np.random.choice(np.arange(len(train_patient_id_list)), int(0.75*len(train_patient_id_list)), replace=False)
            train_patient_id_set = set([train_patient_id_list[idx] for idx in trainset_idx])
            val_patient_id_set = self.train_patient_id_set - train_patient_id_set

            self.train_set = CheXpertGenDataset(root_dir=self.root, patient_id_set=train_patient_id_set, transform=self.transform)
            self.val_set = CheXpertGenDataset(root_dir=self.root, patient_id_set=val_patient_id_set, transform=self.transform)

        if stage == 'test' or stage is None:
            self.test_set = CheXpertGenDataset(root_dir=self.root, patient_id_set=self.test_patient_id_set, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.train_set.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.val_set.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.test_set.collate_fn)

class CheXpertCNN(pl.LightningModule):
    def __init__(self, num_classes, model, lr=5e-5, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
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

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        preds = torch.sigmoid(logits)
        acc = self.val_acc(preds, y.int())  # reuse val_acc for simplicity
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self(x)
        preds = torch.sigmoid(logits)
        return {
            "preds": preds, 
            "target": y
        }

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

def ensemble_pred(models, datamodules, trainer):
    # get predictions for each
    all_preds = []
    all_targets = None

    for m, d in zip(models, datamodules):
        # this returns a list of outputs from your test_step/test_epoch_end
        preds_list = trainer.predict(m, d.test_dataloader())
        # assume preds_list is a list of dicts per batch
        # you’ll need to concatenate them:
        batch_preds = torch.cat([b["preds"] for b in preds_list], dim=0)
        all_preds.append(batch_preds)

        if all_targets is None:
            all_targets = torch.cat([b["target"] for b in preds_list], dim=0)

    # stack & average
    ensemble_preds = torch.stack(all_preds, dim=0).mean(dim=0)

    num_classes = all_targets.shape[-1]
    multilabel_acc = MultilabelAccuracy(num_labels=num_classes, threshold=0.5)
    # ensemble_preds: Tensor of shape (N, C) with probabilities (0–1)
    # all_targets: Tensor of shape (N, C) with 0/1 labels
    acc = multilabel_acc(ensemble_preds, all_targets.int())
    print(f"Ensemble test acc: {acc:.4f}")
    
    return acc