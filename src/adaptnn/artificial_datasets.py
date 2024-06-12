import torch
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy.typing as npt
import numpy as np

import tensorly as tl
import tltorch
from tltorch.functional import convolve as tltorch_convolve

from adaptnn.retina_datasets import SpatioTemporalDatasetBase
from adaptnn.utils import  tuple_convert

from scipy.linalg import orth

class LinearNonlinear(SpatioTemporalDatasetBase):
    def __init__(self,
                 num_time_bins_train : int = 10000,
                 num_time_bins_test : int = 10000,

                 num_cells : int = 8,
                 filter_time: int = 10,
                 filter_spatial : int | tuple[int,int] | None = None, 
                 filter_baseline : float | npt.ArrayLike = 0,
                 filter_rank : int | None = None,
                 out_activation = torch.nn.Softplus,
                 out_noise_std_train : float | npt.ArrayLike | None = 0.1,
                 out_noise_std_test  : float | npt.ArrayLike | None = None,
                 
                 segment_length_bins : int = 100,
                 disjoint_segments : bool = True,
                 device = None,
                 dtype = None,
                 generator : torch.Generator = None
                ):
        
        filter_spatial = tuple_convert(filter_spatial, 2, ())
        assert (len(filter_spatial) in [0,2]), "invalid spatial dimensions given"

        total_time_train = num_time_bins_train + filter_time - 1
        total_time_test  = num_time_bins_test  + filter_time - 1

        with torch.no_grad():
            X_train = torch.randn(filter_spatial + (total_time_train,), generator=generator, dtype=dtype, device=device)
            X_test  = torch.randn(filter_spatial + (total_time_test,), generator=generator, dtype=dtype, device=device)

            if(len(filter_spatial) == 0):
                conv_layer = torch.nn.Conv1d(in_channels=1,out_channels=num_cells,kernel_size=filter_time,device=device, dtype=dtype)
            else:
                conv_layer = torch.nn.Conv3d(in_channels=1,out_channels=num_cells,kernel_size=filter_spatial+(filter_time,),device=device, dtype=dtype)
        
            if(filter_rank is None or filter_rank <= 0 or len(filter_spatial) == 0):
                std = 1/np.sqrt(np.prod(conv_layer.weight.shape[2:]))
                torch.nn.init.normal_(conv_layer.weight,std=std )
            else:
                self.filter_comp_time = orth(np.random.randn(filter_time,filter_rank))
                self.filter_comp_space = orth(np.random.randn(filter_spatial[0]*filter_spatial[1],filter_rank))

                for ii in range(conv_layer.weight.shape[0]):
                    F = (self.filter_comp_time * np.random.randn(1,filter_rank)) @ self.filter_comp_space.T
                    F = F.T.reshape(filter_spatial + (filter_time,))

                    F /= np.sqrt(np.sum(F**2))

                    conv_layer.weight[ii,0,:,:,:] = torch.tensor(F,dtype=dtype,device=device)
                

            conv_layer.bias[:] = filter_baseline

            if(out_activation is None):
                self.model = torch.nn.Sequential(
                            conv_layer,
                            torch.nn.Flatten()
                        )
            else:
                self.model = torch.nn.Sequential(
                            conv_layer,
                            out_activation(),
                            torch.nn.Flatten()
                        )
            
            Y_train = self.model(X_train.unsqueeze(0))
            Y_test  = self.model(X_test.unsqueeze(0))

            if(out_noise_std_train is not None and (out_noise_std_train != 0)):
                Y_train = Y_train + torch.randn(Y_train.shape, generator=generator, dtype=dtype, device=device)*out_noise_std_train
            if(out_noise_std_test is not None and (out_noise_std_test != 0)):
                Y_test = Y_test + torch.randn(Y_test.shape, generator=generator, dtype=dtype, device=device)*out_noise_std_test

            X_train = X_train[...,(filter_time-1):]
            X_test = X_test[...,(filter_time-1):]

        super().__init__(
                X_train,
                X_test,
                Y_train,
                Y_test ,
                normalize_stimulus = False,
                time_padding_bins = filter_time-1,
                segment_length_bins = segment_length_bins,
                disjoint_segments = disjoint_segments)