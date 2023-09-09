import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import h5py

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim

import sys
sys.path.append('../..')

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

import torch
from torch.utils.data import Dataset
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np
import os

class ECGDataset(Dataset):
    def __init__(self, txt_path, ecg_path):
        self.window_size = 2500
        self.fs = 250.0
        self.data = pd.read_csv(txt_path)
        # Drop rows where HashFileName or deid_t_diagnosis_original is NaN
        self.data = self.data.dropna(subset=['HashFileName', 'deid_t_diagnosis_original'])
        self.ecg_path = ecg_path
        self.leads = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']

    def __len__(self):
        return len(self.data)

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
            x_new = np.linspace(0, 1, num=t*fs_out, endpoint=True)
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
        row = self.data.iloc[idx]
        hash_file_name = row['HashFileName']
        diagnosis = row['deid_t_diagnosis_original']

        s_dirs = [f"S{i:04d}" for i in range(1, 5)] # Assuming there are only 4 'S' directories, modify as needed
        year_dirs = [str(i) for i in range(1987, 2024)] # Assuming years range from 1980 to 2020
        month_dirs = [f"{i:02}" for i in range(1, 13)]
        try:
        # Iterate over all possible combinations to find the file
          file_found = False
          for s_dir in s_dirs:
              for year_dir in year_dirs:
                  for month_dir in month_dirs:
                      file_path = f"{self.ecg_path}/{s_dir}/{year_dir}/{month_dir}/{hash_file_name}"
                      if os.path.exists(file_path):
                          hd5_file = h5py.File(file_path, "r")
                          file_found = True
                          break
                  if file_found:
                      break
              if file_found:
                  break
          #hd5_file = h5py.File(f"{self.ecg_path}/{hash_file_name}", "r")
          for k in list(hd5_file['ecg'].keys()):
            ecg_data_list = [torch.tensor(hd5_file['ecg'][k][lead][:]) for lead in self.leads]
            ecg_data = torch.stack(ecg_data_list, dim=0)
            sample_rate = float(hd5_file['ecg'][k]['ecgsamplebase_pc'][()])
        
    
          ecg_data = np.array(ecg_data, dtype=np.float64)
          ecg_data = self.preprocess(ecg_data, sample_rate)
          ecg_data = torch.tensor(ecg_data, dtype=torch.float)

          sample = {'ecg': ecg_data, 'txt': diagnosis }
          return sample
    
        except Exception as e:
          print(f"Error reading file {file_path}: {e}")
          normal_file_path = '/home/ubuntu/data/ecg/S0003/2010/02/de_115828429_20111219012630_20111219114122.hd5'
          row1 = self.data.iloc[2]
          diagnosis1 = row1['deid_t_diagnosis_original']
          hd5_file1 = h5py.File(normal_file_path, "r")
          for k in list(hd5_file1['ecg'].keys()):
            ecg_data_list = [torch.tensor(hd5_file1['ecg'][k][lead][:]) for lead in self.leads]
            ecg_data1 = torch.stack(ecg_data_list, dim=0)
            sample_rate1 = float(hd5_file1['ecg'][k]['ecgsamplebase_pc'][()])
        
    
            ecg_data1 = np.array(ecg_data1, dtype=np.float64)
            ecg_data1 = self.preprocess(ecg_data1, sample_rate1)
            ecg_data1 = torch.tensor(ecg_data1, dtype=torch.float)

            sample1 = {'ecg': ecg_data1, 'txt': diagnosis1 }
            return sample1
        
          #if not file_found:
          #    raise ValueError(f"File {self.ecg_path}/{s_dir}/{year_dir}/{month_dir}/{hash_file_name} not found in the directory structure.")
        
        # Load the corresponding hd5 file for ECG data


def load_data(ecg_path, txt_path, batch_size=128, column='deid_t_diagnosis_original'): 
    if torch.cuda.is_available():  
        dev = "cuda:0" 
        cuda_available = True
        print('Using CUDA.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using CPU.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)
    
    torch_dset = ECGDataset(ecg_path=ecg_path, txt_path=txt_path)
    data_loader = data.DataLoader(torch_dset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    
    return data_loader, device
    
# Testing the dataset with the provided files
#cxr_filepath = '/home/ubuntu/data'
#txt_filepath = '/home/ubuntu/data/sample.csv'
#dataset = ECGDataset(txt_filepath, cxr_filepath)
#ECGdataloader,dev = load_data(txt_path = txt_filepath, ecg_path = cxr_filepath)

    
def load_clip(model_path=None, pretrained=False, context_length=77):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a model with the CLIP model 
    architecture. 
    
    args: 
        * model_path (optional) - path to model weights that the model
        will be initialized with 
        * pretrained (optional) - if True, will load the pretrained 
        CLIP model
        * context_length (optional) - length of the maximum number of 
        tokens that can be inputted into the CLIP model
    '''

    params = {
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    }
    
    # set device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if pretrained: 
        # load clip pre-trained model
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
    else: 
        model = CLIP(**params)
        print("Loaded in clip model.")
    
    # if a model_path is provided, load in weights to backbone
    if model_path != None: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model
    
    
def preprocess_text(texts, model):
#     if model.context_length is None: 
#         model = model.module
        
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)

    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[:model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

"""
def make(config, cxr_filepath, txt_filepath, model_path=None): 
    '''
    FUNCTION: make
    ---------------------------------
    This function makes the model, the data loader, loss and optimizer. 
    
    args: 
        * config - dict, configuration of experiment
        * cxr_filepath - string, filepath to chest x-ray images
        * txt_filepath - string, filepath to corresponding text reports
        * model_path - string, filepath to previously trained model
    '''
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=config.batch_size, pretrained=config.pretrained, column=config.column)
    model = load_clip(model_path=model_path, pretrained=config.pretrained, context_length=config.context_length)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    # todo: incorporate - torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    return model, data_loader, device, criterion, optimizer


def train_main(cxr_filepath, txt_filepath, hyperparams, output_path, model_path=None, pretrained=False): 
    '''
    args: 
        * cxr_filpath- str filepath to cxr images
        * txt_filepath- str filepath to text reports
        * hyperparams- dictionary with the following hyperparams:
        `batch_size`, `criterion`, `learning_rate`, `momentum`, `epochs`
        * output_path- str filepath to where the trained model will be saved
        * model_path- str filepath to model that will be used as baseline model for training. 
        If not provided, a model will be trained from scratch
        * pretrained- whether or not the clip model was pretrained with generic images 
    This function is the main train function for CXR-CLIP. 
    '''
    
    # unpack `hyperparams`
    batch_size = hyperparams['batch_size']
    criterion = hyperparams['criterion']
    learning_rate = hyperparams['learning_rate']
    momentum = hyperparams['momentum']
    epochs = hyperparams['epochs']
    
    # load input cxr + report data
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=batch_size, pretrained=pretrained)
    model = load_clip(model_path=model_path, pretrained=pretrained)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_clip(model, data_loader, device, criterion, optimizer, epochs, output_path)
    return model
"""