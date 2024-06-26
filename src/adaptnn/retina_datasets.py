import h5py
from scipy.io import loadmat
import numpy as np
import torch
import warnings
from adaptnn.utils import  tuple_convert


class SpatioTemporalDatasetBase(torch.utils.data.Dataset):
    def __init__(self,
                 X_train : torch.Tensor, # (X,Y,T)
                 X_test  : torch.Tensor, # (N,T)
                 Y_train : torch.Tensor,
                 Y_test  : torch.Tensor,
                 normalize_stimulus : bool,
                 time_padding_bins : int,
                 segment_length_bins : int,
                 disjoint_segments : bool
                ):
        self.time_padding_bins_ = None
        self.segment_length_bins_ = segment_length_bins

        self.X_sig, self.X_mu = torch.std_mean(X_train,dim=-1,keepdim=True)
       
        self.normalize_stimulus = normalize_stimulus
        if(normalize_stimulus):
            X_train = self.transform_X(X_train)
            X_test = self.transform_X(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        

        self.disjoint_segments_ = disjoint_segments
        self.time_padding_bins = time_padding_bins

        self.reset_start_indices_()
    @property 
    def segment_length_bins(self) -> int:
        if(self.segment_length_bins_ is None):
            self.segment_length_bins_ = 100
        return self.segment_length_bins_
    @segment_length_bins.setter
    def segment_length_bins(self, segment_length_bins_ : int) -> None:
        if(self.segment_length_bins_ == segment_length_bins_):
            return
        self.segment_length_bins_ = segment_length_bins_
        self.reset_start_indices_()

    @property
    def disjoint_segments(self) -> str | bool:
        return self.disjoint_segments_
    @disjoint_segments.setter
    def disjoint_segments(self, disjoint_segments_ : str | bool) -> None:
        if(self.disjoint_segments_ == disjoint_segments_):
            return
        self.disjoint_segments_ = disjoint_segments_
        self.reset_start_indices_()


    @property
    def time_padding_bins(self) -> int:
        if(self.time_padding_bins_ is None):
            self.time_padding_bins_ = 0
        return self.time_padding_bins_
    


    @time_padding_bins.setter
    def time_padding_bins(self, time_padding_bins_ : int) -> None:
        '''
        Sets how many time points are needed in an input preceding the output (this is time series data)
        '''
        if(self.time_padding_bins_ == time_padding_bins_):
            return
        
        self.time_padding_bins_ = time_padding_bins_
        self.reset_start_indices_()

    def reset_start_indices_(self) -> None:
        self.X_time = self.time_padding_bins + self.segment_length_bins
        if(self.disjoint_segments == 'full'):
            # completely disconnects segments: otherwise time padding period can be the end of another segment
            self.start_idx_X_train = torch.arange(0, self.X_train.shape[-1]-self.X_time+1, self.X_time)
            self.start_idx_Y_train = self.start_idx_X_train + self.time_padding_bins
        elif(self.disjoint_segments):
            self.start_idx_X_train = torch.arange(0, self.X_train.shape[-1]-self.X_time+1, self.segment_length_bins)
            self.start_idx_Y_train = self.start_idx_X_train + self.time_padding_bins
        else:
            self.start_idx_X_train = torch.arange(0, self.X_train.shape[-1]-self.X_time+1)
            self.start_idx_Y_train = self.start_idx_X_train + self.time_padding_bins

        last_bin = (self.start_idx_X_train.max().cpu().numpy() + self.X_time) 
        first_bin = self.start_idx_X_train.min().cpu().numpy()

        unused_bins = (self.X_train.shape[-1] - (last_bin+1)) + first_bin

        if(unused_bins > 0):
            warnings.warn(f"Segmentation of dataset leaves {unused_bins} time bins unseen.")
        if(last_bin >=  self.X_train.shape[-1] or first_bin < 0):
            raise RuntimeError("Invalid segmentation of dataset: will return invalid indices.")

    def __len__(self) -> int:
        return len(self.start_idx_X_train)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        size (1,X,Y,T+time_padding_bins), (num_cells,T)
        '''
        start_x = self.start_idx_X_train[idx]
        end_x   = start_x + self.X_time

        start_y = self.start_idx_Y_train[idx]
        end_y   = start_y + self.segment_length_bins
        return self.X_train[...,start_x:end_x].unsqueeze(0), self.Y_train[:,start_y:end_y]
    
    def get_test(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns:
            tuple (test input,test output) of sizes (1,1,X,Y,T+time_padding_bins),(1,num_cells,T)
        '''
        return self.X_test.unsqueeze(0).unsqueeze(0), self.Y_test[:,self.time_padding_bins:].unsqueeze(0)
    
    @property
    def temporal_only(self) -> bool:
        return self.X_train.ndim != 3

    @property
    def frame_width(self) -> int:
        if(self.X_train.ndim == 3):
            return self.X_train.shape[0]
        else:
            return None
        
    @property
    def frame_height(self) -> int:
        if(self.X_train.ndim == 3):
            return self.X_train.shape[1]
        else:
            return None
    
    def transform_X(self, xx : torch.Tensor) -> None:
        '''
        Normalizes a video input (...,X,Y,T) pixelwise given the fitted X_sig and X_mu parameters.
        '''
        if(not self.normalize_stimulus):
            return xx
        else:
            return (xx - self.X_mu)/self.X_sig 
     
    @property
    def num_cells(self) -> int:
        return self.Y_train.shape[0]


class NiruDataset(SpatioTemporalDatasetBase):
    '''
    Loads up one dataset from

    Maheswaranathan, Niru, Lane T. McIntosh, Hidenori Tanaka, Satchel Grant, David B. Kastner,
    Joshua B. Melander, Aran Nayebi et al. "Interpreting the retinal neural code for natural scenes:
    From computations to neurons." Neuron 111, no. 17 (2023): 2742-2755.


    The data can be downloaded from the link given in the paper.

    Breaks the training data up into segments of a given length for fitting with batches.
    '''
    def __init__(self, 
                 time_padding_bins : int = 50,
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


        self.data_file = data_root + fname
        f = h5py.File(self.data_file,"r")

        X_train_local = np.array(f["train"]["stimulus"])
        X_test_local  = np.array(f["test"]["stimulus"])
        Y_train_local = np.array(f["train"]["response"][response_type_train])
        Y_test_local  = np.array(f["test"]["response"][response_type_test])

        X_train = torch.tensor(X_train_local, device=device, dtype=dtype)
        X_train = X_train.moveaxis(0,2)
        X_test  = torch.tensor(X_test_local,  device=device, dtype=dtype)
        X_test  = X_test.moveaxis(0,2)
        Y_train = torch.tensor(Y_train_local, device=device, dtype=dtype)
        Y_test  = torch.tensor(Y_test_local, device=device, dtype=dtype)
        
        self.recording = recording
        self.response_type_train = response_type_train
        self.response_type_test = response_type_test

        super().__init__(X_train,
                 X_test,
                 Y_train,
                 Y_test,
                 normalize_stimulus,
                 time_padding_bins,
                 segment_length_bins,
                 disjoint_segments)




class MultiContrastFullFieldJN05Dataset(torch.utils.data.Dataset):
    '''
    Currently only bins spikes at the frame length level - no upsampling stimulus yet.
    '''
    def __init__(self,
                 time_padding_bins : int = 50,
                 train_long_contrast_levels : int | tuple[int] = (1,3),
                 test_long_contrast_levels  : int | tuple[int] = (1,2,3),
                 test_rpt_contrast_levels   : int | tuple[int] = (1,2,3),
                 train_long_period : tuple[int,int] = (1000,71000),
                 test_long_period : tuple[int,int]  = (75000,95000),
                 segment_length_bins : int = 240,
                 disjoint_segments : bool = True,
                 device = None,
                 dtype = None,
                 base_dir : str = "/media/latimerk/ExtraDrive1/cbem/Data/JN05/flashesRGC_JN05/"):
        self.base_dir = base_dir 
        self.time_padding_bins_ = None

        self.frame_length = 0.00834072 # in s
        with open(f"{self._full_dir(0)}/framelen") as f:
            data = f.read()
            self.frame_length = float(data) # should be the same as above, but reloading to show how dataset info was originally stored 

        self.train_long_contrast_levels = tuple_convert(train_long_contrast_levels)
        self.train_long_period = tuple_convert(train_long_period)
        assert len(self.train_long_period) == 2, "segment period must be a tuple of length 2"
        assert self.train_long_period[1]>self.train_long_period[0], "train_long_period[1] must be greater than train_long_period[0]"
        assert len(self.train_long_contrast_levels)>=1, "no train_long_contrast_levels given."

        self.test_rpt_contrast_levels = tuple_convert(test_rpt_contrast_levels)
        if(test_long_contrast_levels is not None):
            test_long_contrast_levels = tuple_convert(test_long_contrast_levels)
        if(test_long_period is not None):
            test_long_period = tuple_convert(test_long_period)

        self.test_long_period  = test_long_period
        self.test_long_contrast_levels = test_long_contrast_levels
        self.test_long = (test_long_period is not None) and ((test_long_contrast_levels is not None) and (len(test_long_contrast_levels) > 0))
        if(self.test_long ):
            assert len(self.test_long_period) == 2, "segment period must be a tuple of length 2"
            assert self.test_long_period[1]>self.test_long_period[0], "test_long_period[1] must be greater than test_long_period[0]"

        self.long_contrasts = tuple(set(test_long_contrast_levels).union(train_long_contrast_levels))
        self.long_contrast_index = {}

        self.Y_full = None
        self.X_full = None
        for ci, contrast in enumerate(self.long_contrasts):
            self.long_contrast_index[contrast] = ci

            X_c, Y_c = self._load_long_recording(contrast)
            if(self.Y_full is None):
                self.Y_full = np.zeros((len(self.long_contrasts),) + Y_c.shape)
            if(self.X_full is None):
                self.X_full = np.zeros((len(self.long_contrasts),) + X_c.shape)
            self.X_full[ci,...] = X_c
            self.Y_full[ci,...] = Y_c

        X_rpt = None
        Y_rpt = None
        self.rpt_contrast_index = {}
        for ci, contrast in enumerate(self.test_rpt_contrast_levels):
            self.rpt_contrast_index[contrast] = ci

            X_c, Y_c = self._load_rpt_recording(contrast)
            if(Y_rpt is None):
                Y_rpt = np.zeros((len(self.test_rpt_contrast_levels),) + Y_c.shape)
            if(X_rpt  is None):
                X_rpt = np.zeros((len(self.test_rpt_contrast_levels),) + X_c.shape)
            X_rpt[ci,...] = X_c
            Y_rpt[ci,...] = Y_c
        self.X_rpt_0 = torch.tensor(X_rpt, device=device, dtype=dtype)
        self.Y_rpt = torch.tensor(Y_rpt, device=device)

        self.dtype=dtype
        self.device=device

        self.disjoint_segments = disjoint_segments 
        self.segment_length_bins = segment_length_bins
        self.X_time = time_padding_bins + self.segment_length_bins
        self.start_idx_X_train = np.array([0])
        self.start_idx_Y_train = self.start_idx_X_train + time_padding_bins

        self.time_padding_bins = time_padding_bins

    def _get_num_repeats(self):
        num_rpts = None
        for contrast in self.test_rpt_contrast_levels:
            fname_rpt_spks = f"{self._full_dir(contrast)}/Mtsp_rpt.mat"
            f_spks = loadmat(fname_rpt_spks)
            mtsp = f_spks["Mtsp_rpt"].ravel()
            num_rpts_c = min([cc.shape[1] for cc in mtsp])
            if(num_rpts is None or num_rpts_c < num_rpts):
                num_rpts = num_rpts_c
        return num_rpts

    def _load_rpt_recording(self, contrast : int) -> tuple[np.ndarray,np.ndarray] :
        fname_rpt_stim = f"{self._full_dir(contrast)}/Stim_rpt.mat"
        fname_rpt_spks = f"{self._full_dir(contrast)}/Mtsp_rpt.mat"

        f_spks = loadmat(fname_rpt_spks)
        f_stim = loadmat(fname_rpt_stim)

        stim = f_stim["Stim"].ravel()
        mtsp = f_spks["Mtsp_rpt"].ravel()

        T = stim.size;
        num_cells = mtsp.size
        num_rpts = self._get_num_repeats()

        Y = np.zeros((num_rpts,num_cells,T),dtype=int)
        for cell_num,spks in enumerate(mtsp):
            for rpt in range(num_rpts):
                spks = mtsp[cell_num][:,rpt] - 1
                Y[rpt,cell_num,:],_ = np.histogram(spks-1, bins=np.arange(0,T+1))

        return stim, Y

    def _load_long_recording(self, contrast : int) -> tuple[np.ndarray,np.ndarray]:
        fname_stim = f"{self._full_dir(contrast)}/Stim_long.mat"
        fname_spks = f"{self._full_dir(contrast)}/Mtsp_long.mat"

        f_spks = loadmat(fname_spks)
        f_stim = loadmat(fname_stim)

        stim = f_stim["Stim"].ravel()
        mtsp = f_spks["Mtsp"].ravel()

        T = stim.size;

        Y = np.zeros((mtsp.size,T),dtype=int)

        for cell_num,spks in enumerate(mtsp):
            spks = mtsp[cell_num] - 1
            Y[cell_num,:],_ = np.histogram(spks-1, bins=np.arange(0,T+1))

        return stim, Y

    def __len__(self) -> int:
        return len(self.start_idx_X_train) * len(self.train_long_contrast_levels)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        size (1,T+time_padding_bins), (num_cells,T)
        '''
        ci = idx//len(self.start_idx_X_train)
        idx_c = idx % len(self.start_idx_X_train)

        start_x = self.start_idx_X_train[idx_c]
        end_x   = start_x + self.X_time

        start_y = self.start_idx_Y_train[idx_c]
        end_y   = start_y + self.segment_length_bins
        return self.X_train[ci,:,start_x:end_x], self.Y_train[ci,:,start_y:end_y]

    def _full_dir(self,contrast):
        contrast_suffixes = {0 : "", 1 : "b", 2 : "c", 3 : "d"}
        return f"{self.base_dir}/flash2001-04-06{contrast_suffixes[contrast]}"
    

    @property
    def time_padding_bins(self) -> int:
        if(self.time_padding_bins_ is None):
            self.time_padding_bins_ = 0
        return self.time_padding_bins_
    
    @time_padding_bins.setter
    def time_padding_bins(self, time_padding_bins_ : int) -> None:
        '''
        Sets how many time points are needed in an input preceding the output (this is time series data).
        This also loads the data segments to the device
        '''
        if(self.time_padding_bins == time_padding_bins_):
            return

        self.time_padding_bins_ = time_padding_bins_
        self.X_time = time_padding_bins_ + self.segment_length_bins


        ci_train = [self.long_contrast_index[xx] for xx in self.train_long_contrast_levels]
        x_start_train = self.train_long_period[0] - time_padding_bins_
        self.X_train = torch.tensor(self.X_full[ci_train, x_start_train:self.train_long_period[1]],
                                    device=self.device, dtype=self.dtype).unsqueeze(1)
        self.Y_train = torch.tensor(self.Y_full[ci_train, :,self.train_long_period[0]:self.train_long_period[1]],
                                    device=self.device)

        if(self.disjoint_segments == 'full'):
            # completely disconnects segments: otherwise time padding period can be the end of another segment
            self.start_idx_X_train = torch.arange(0, self.X_train.shape[-1]-self.X_time+1, self.X_time)
            self.start_idx_Y_train = self.start_idx_X_train 
        elif(self.disjoint_segments):
            self.start_idx_X_train = torch.arange(0, self.X_train.shape[-1]-self.X_time+1, self.segment_length_bins)
            self.start_idx_Y_train = self.start_idx_X_train 
        else:
            self.start_idx_X_train = torch.arange(0, self.X_train.shape[-1]-self.X_time+1)
            self.start_idx_Y_train = self.start_idx_X_train

        if(self.test_long):
            ci_test = [self.long_contrast_index[xx] for xx in self.test_long_contrast_levels]
            x_start_test= self.train_long_period[0] - time_padding_bins_
            self.X_test = torch.tensor(self.X_full[ci_test, x_start_test:self.test_long_period[1]],
                                        device=self.device, dtype=self.dtype).unsqueeze(1)
            self.Y_test = torch.tensor(self.Y_full[ci_test, :, self.test_long_period[0]:self.test_long_period[1]],
                                        device=self.device)
            

        
        shape_rpt_padding = list(self.X_rpt_0.shape)
        shape_rpt_padding[1] = time_padding_bins_
        self.X_rpt = torch.concat([torch.zeros(shape_rpt_padding,
                                                device=self.X_rpt_0.device,
                                                dtype=self.X_rpt_0.dtype),
                                    self.X_rpt_0
                                   ], 
                                   dim=1).unsqueeze(1)
        
    
    def get_test_long(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns:
            tuple (test input,test output) of sizes (N_test_contrast,1,T+time_padding_bins),(N_test_contrast,num_cells,T)
        '''
        if(self.test_long):
            return self.X_test, self.Y_test
        else:
            raise ValueError("no long test conditions loaded")
        
    def get_test_rpt(self,contrast : int) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns:
            tuple (test input,test output) of sizes (1,T+time_padding_bins),(n_rpts,num_cells,T)
        '''
        ci = self.rpt_contrast_index[contrast] 
        return self.X_rpt[ci,...], self.Y_rpt[ci,...]
    

    @property
    def num_cells(self) -> int:
        return self.Y_full.shape[1]