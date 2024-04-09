import h5py
import torch
import numpy as np
from sklearn import preprocessing

class NiruDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 time_padding_bins : int,
                 segment_length_bins : int = 100,
                 disjoint_segments : bool = True,
                 recording : str ="A-natural",
                 response_type_train : str = "binned",
                 response_type_test : str = "firing_rate_10ms",
                 device = None,
                 dtype = None,
                 data_root : str = "data/ganglion_cell_data/",
                 normalize_stimulus : bool = True
                ):
        if(recording == "A-natural"):
            fname ="15-11-21a/naturalscene.h5"
        elif(recording == "A-noise"):
            fname ="15-11-21a/whitenoise.h5"
        elif(recording == "B-noise"):
            fname ="15-11-21b/whitenoise.h5"
        else:
            raise ValueError("Unknown dataset")

        f = h5py.File(data_root + fname,"r")
        self.data_file = data_root + fname

        self.X_train = torch.tensor(f["train"]["stimulus"], device=device, dtype=dtype)
        self.X_train = self.X_train.transpose(0,2)
        self.X_train = self.X_train.transpose(2,1)
        self.X_test = torch.tensor(f["test"]["stimulus"], device=device, dtype=dtype)
        self.X_test = self.X_test.transpose(0,2)
        self.X_test = self.X_test.transpose(2,1)

        self.Y_train = torch.tensor(f["train"]["response"][response_type_train], device=device, dtype=dtype)
        self.Y_test = torch.tensor(f["test"]["response"][response_type_test], device=device, dtype=dtype)

        self.X_sig, self.X_mu = torch.std_mean(self.X_train,dim=-1,keepdim=True)

        self.X_train = self.transform_X(self.X_train)
        self.X_test = self.transform_X(self.X_test)
        

        self.disjoint_segments = disjoint_segments
        self.recording = recording
        self.response_type_train = response_type_train
        self.response_type_test = response_type_test

        self.normalize_stimulus = normalize_stimulus
        
        self.segment_length_bins = segment_length_bins
        self.time_padding_bins = time_padding_bins
        self.X_time = time_padding_bins + segment_length_bins

        if(self.disjoint_segments):
            self.start_idx_X_train = torch.range(self.X_time, self.X_train.shape[-1]-self.X_time, self.segment_length_bins)
            self.start_idx_Y_train = self.start_idx_X_train + time_padding_bins
        else:
            self.start_idx_X_train = torch.range(self.X_time, self.X_train.shape[-1]-self.X_time)
            self.start_idx_Y_train = self.start_idx_X_train + time_padding_bins

    def set_timepadding(self, time_padding_bins):
        self.time_padding_bins = time_padding_bins
        self.X_time = time_padding_bins + self.segment_length_bins

        if(self.disjoint_segments):
            self.start_idx_X_train = torch.range(self.X_time, self.X_train.shape[-1]-self.X_time, self.segment_length_bins)
            self.start_idx_Y_train = self.start_idx_X_train + time_padding_bins
        else:
            self.start_idx_X_train = torch.range(self.X_time, self.X_train.shape[-1]-self.X_time)
            self.start_idx_Y_train = self.start_idx_X_train + time_padding_bins

    def transform_X(self, xx):
        if(not self.normalize_stimulus):
            return xx
        else:
            return (xx - self.X_mu)/self.X_sig
        
        

    def __len__(self):
        return len(self.start_idx_X_train)
    
    def __getitem__(self, idx) :
        start_x = self.start_idx_X_train[idx]
        end_x   = start_x + self.X_time

        start_y = self.start_idx_Y_train[idx]
        end_y   = start_y + self.segment_length_bins
        return self.X_train[:,:,start_x:end_x], self.Y_train[:,start_y:end_y]
    

    def get_test(self):
        return self.X_test.unsqueeze(0), self.Y_test[:,self.time_padding_bins:].unsqueeze(1)
    
    @property
    def frame_width(self):
        return self.X_train.shape[0]
    @property
    def frame_height(self):
        return self.X_train.shape[1]
    @property
    def num_cells(self):
        return self.Y_train.shape[0]