import subprocess
import numpy as np
import os
import sys
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from typing import List, Tuple
from torch.utils.data import Dataset
import torch
from torch.utils import data
from tqdm.notebook import tqdm
import torch.nn as nn
from scipy.interpolate import interp1d

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score

from model import CLIP
import clip
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score


class ECGDataset(Dataset):
    def __init__(self, data_path, label_path):
        """
        Args:
            data_path (string): 路径到心电图数据的 .npy 文件。
            label_path (string): 路径到标签数据的 .npy 文件。
        """
        self.window_size = 2500
        self.fs = 250.0
        self.data = np.load(data_path)
        self.labels = np.load(label_path).squeeze()

        assert self.data.shape[0] == self.labels.shape[0], \
            "Data and labels must have the same number of samples!"

    def __len__(self):
        return self.data.shape[0]
    def resample_unequal(self, ts, fs_in, fs_out):
        if fs_in == 0:
            return ts
        t = len(ts) / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)
        if fs_out == fs_in:
            return ts
        if 2*fs_out == fs_in:
            return ts[::2]
        else:
            x_old = np.linspace(0, 1, num=len(ts), endpoint=True)
            x_new = np.linspace(0, 1, num=int(t*fs_out), endpoint=True)

            y_old = ts
            f = interp1d(x_old, y_old, kind='linear')
            y_new = f(x_new)
            return y_new
    
    def preprocess(self, arr, sample_rate):
        """
        arr has shape (n_channel, n_length)

        """
        out = []
        for tmp in arr:

            # resample
            if sample_rate != self.fs:
                tmp = self.resample_unequal(tmp, sample_rate, self.fs)

            # filter
            # tmp = notch_filter(tmp, self.fs, 60, verbose='ERROR')
            # tmp = filter_data(tmp, self.fs, 0.5, 50, verbose='ERROR')

            out.append(tmp)

        out = np.array(out)
        n_length = out.shape[1]

        if n_length > self.window_size: # crop center window_size for longer
            i_start = (n_length-self.window_size)//2
            i_end = i_start+self.window_size
            out = out[:, i_start:i_end]
        elif n_length < self.window_size: # pad zeros for shorter
            pad_len = np.zeros((len(self.leads), self.window_size-n_length))
            out = np.concatenate([out, pad_len], axis=1)

        return out
    def __getitem__(self, idx):
        ecg = torch.tensor(self.data[idx], dtype=torch.float32) * 200
        ecg = ecg.T
        ecg = np.array(ecg, dtype=np.float64)
        ecg = self.preprocess(ecg, sample_rate=100)
        ecg = torch.tensor(ecg, dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        sample = {'ecg': ecg, 'label':label}
        return sample

def load_clip(model_path, pretrained=False, context_length=77): 
    """
    FUNCTION: load_clip
    ---------------------------------
    """
    device = torch.device("cuda:0")
    if pretrained is False: 
        # use new model params
        params = {
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
        }

        model = CLIP(**params)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    else: 
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False) 
    try: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    except: 
        print("Argument error. Set pretrained = True.", sys.exc_info()[0])
        raise
    return model

def make(
    model_path: str, 
    cxr_filepath: str, 
    pretrained: bool = False, 
    context_length: bool = 77, 
):
    """
    FUNCTION: make
    -------------------------------------------
    This function makes the model, the data loader, and the ground truth labels. 
    
    args: 
        * model_path - String for directory to the weights of the trained clip model. 
        * context_length - int, max number of tokens of text inputted into the model. 
        * cxr_filepath - String for path to the chest x-ray images. 
        * cxr_labels - Python list of labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * pretrained - bool, whether or not model uses pretrained clip weights
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.
    
    Returns model, data loader. 
    """
    # load model
    model = load_clip(
        model_path=model_path, 
        pretrained=False, 
        context_length=context_length
    )

    
    # create dataset
    torch_dset = ECGDataset(data_path="/home/ubuntu/code/ECG2TEXT/X_all.npy", label_path="/home/ubuntu/code/ECG2TEXT/y_test.npy")
    
    loader = torch.utils.data.DataLoader(torch_dset, batch_size=16, shuffle=False)
    
    return model, loader

model_dir = '/home/ubuntu/code/ECG2TEXT/checkpoints/pt-imp/checkpoint_28000.pt'
cxr_filepath = "/home/ubuntu/code/ECG2TEXT/X_all.npy"
model, loader = make(model_path = model_dir, cxr_filepath = cxr_filepath)

#imagenet_classes = ["Normal ECG","Myocardial Infarction","ST/T change","Conduction Disturbance", "Hypertrophy"]

imagenet_classes = ["NORMAL","INFARCT"," ST T","BRANCH BLOCK", "HYPERTROPHY ENLARGEMENT"]

imagenet_templates = [
    #'ECG IS {}.'
    '{}'
]

print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            
            texts = [template.format(classname) for template in templates] #format with class\
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def multi_class_roc_auc_avg(logits, target):
    
    num_classes = logits.shape[1]
    roc_auc_scores = []

    for c in range(num_classes):
        # Convert multi-class labels to binary labels
        binary_labels = (target == c).astype(int)
        
        # Check if there is at least one positive and one negative sample
        if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
            score = roc_auc_score(binary_labels, logits[:, c])
            roc_auc_scores.append(score)
        else:
            roc_auc_scores.append(float('nan'))
    
    # Compute the average ROC-AUC excluding NaN values
    avg_roc_auc = np.nanmean(roc_auc_scores)
            
    return avg_roc_auc,roc_auc_scores

with torch.no_grad():
    top1, top5, n, meanauc = 0., 0., 0., 0.
    for data in tqdm(loader):
        images = data['ecg']
        target = data['label']
        images = images.cuda()
        target = target.cuda()


        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ zeroshot_weights
        #print(logits)
        #print(target)
        logits_np = logits.cpu().numpy()
        target_np = target.cpu().numpy()

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        mean_auc,roc_auc_scores = multi_class_roc_auc_avg(logits_np, target_np)
        #meanauc += mean_auc
        #roc_auc_scores += roc_auc_scores
        top1 += acc1
        top5 += acc5
        n += images.size(0)
        #if n % 100 == 0:
          #print(n)
top1 = (top1 / n) * 100
print(f"Top-1 accuracy: {top1:.2f}")
print(f"mean_auc: {mean_auc:.2f}")
#print(roc_auc_scores)