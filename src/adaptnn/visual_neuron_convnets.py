import torch
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy.typing as npt
import numpy as np

import tensorly as tl
import tltorch
from tltorch.functional import convolve as tltorch_convolve

from adaptnn.utils import compute_conv_output_size, tuple_convert

'''
===============================================================================
===============================================================================
Specialized linear/conv layers
===============================================================================
===============================================================================
'''

class TransposeLinear(torch.nn.Module):
    '''
    Special version of a linear layer with an additional reshape/transpose
    useful for conv layers into fully connected over time in the time series models I'm interested in for fitting RGC responses.

    Takes input of size (..., in_features, K) where in_features can be a tuple.
    Transforms it into (..., K, total_in_features) where total_in_features = prod(in_features)
    And then does a linear layer with input size total_in_features and output size out_features
    Transposes again to return output of size
    (..., output_features, K)

    The linear weights A is a matrix of size (total_in_features, output_features)
    '''
    def __init__(self, in_features : Tuple | int, out_features : int, **kwargs):
        '''
        in_features : int or tuple of ints. The (multi-dimensional) input size. If multiple dimensions, these will be flattened.
                        The total number of input features that will go into the Linear layer is prod(in_features).
        out_features: the number of output features (the same as torch.nn.Linear) 
        kwargs: keyword arguments for the torch.nn.Linear layer.
        '''

        super().__init__()
        in_features = tuple_convert(in_features)

        self.in_features = in_features;
        self.in_features_total = np.prod(in_features)
        self.n_in_features = len(in_features)

        self.linear = torch.nn.Linear(self.in_features_total, out_features, **kwargs)


    def forward(self, X):
        if(self.n_in_features > 1):
            # for flattening results of a Conv3D conv into the right fully connected layer
            X = torch.nn.Flatten(start_dim=-1-self.n_in_features, end_dim=-2);
        
        # flips dimensions
        X = torch.transpose(X, dim0=-2, dim1=-1)
        # linear op
        X = self.linear(X)
        # flips dimensions back: idea is to get a (N_samples, Neuron, Time) output tensor
        X = torch.transpose(X, dim0=-2, dim1=-1)
        return X    

    @property
    def weight(self):
        return self.linear.weight        
    
    def __repr__(self):
        msg = (f'{self.__class__.__name__} wrapping {self.linear.__repr__()}')
        return msg

class SpatioTemporalTuckerRFConv3D(torch.nn.Module):
    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : tuple, rank : int,
                 factorization_type = 'spatial',
                 bias : bool = True,
                 stride=1, padding=0, dilation=1, groups=1,
                 device=None, dtype=None):
        '''
        Effectively an altered version of a 3D tltorch.FactorizedConv layer with reshaping of space in the factorized weights to try to avoid some overparameterization.
        Args:
            factorization_type : {'spatial', 'spatialchannel'} How the spatial factors are parameterized.
                                If 'spatial' = The x & y spatial factors combined into vectors of length (x*y). This avoids needing full-rank x,y components
                                If 'spatialchannel' = The x & y spatial factors and input channels combined into vectors of length (x*y*in_channels).
            rank: The Tucker rank of the weights given the factorization used. For factorization_type='spatial', the rank is (out_channel rank, in_channel rank, spatial rank, temporal rank)
            The remaining arguments are just as for a torch.nn.Conv3D
        '''
        
        super().__init__()

        if(factorization_type is None):
            factorization_type = 'spatial'

        kernel_size = tuple_convert(kernel_size)
        assert len(kernel_size) == 3, "kernel size must be length 3"

        if(groups != 1):
            raise NotImplementedError("Groups can only be 1 for Tucker Tensorized Conv layers.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank

        self.space_channels  =  kernel_size[0] * kernel_size[1]
        self.in_group_channels = (in_channels//groups);

        self.weights_shape = (out_channels, self.in_group_channels) +  kernel_size

        self.factorization_type = factorization_type
        if(self.factorization_type == 'spatial'):
            self.factorization_shape = (out_channels, self.in_group_channels, kernel_size[0]*kernel_size[1], kernel_size[2])
            rank = list(tuple_convert(rank, 4))
            assert len(rank) == len(self.factorization_shape), "rank must be shape (out_channels, in_channels, space, time)"
        elif(self.factorization_type == 'spatialchannel'):
            self.factorization_shape = (out_channels, self.in_group_channels * kernel_size[0]*kernel_size[1], kernel_size[2])
            rank = list(tuple_convert(rank, 3))
            assert len(rank) == len(self.factorization_shape), "rank must be shape (out_channels, in_channels&space, time)"
        else:
            raise ValueError(f"Invalid factorization_type: {factorization_type}")
        
        # make sure ranks are not too large
        for xxi, xx in enumerate(self.factorization_shape):
            rank[xxi] = min(rank[xxi],xx);
        rank = tuple(rank)

        # self.weights_space_ = torch.zeros(out_channels, in_channels/groups, kernel_size[0], kernel_size[1], rank)
        self.weight_tucker = tltorch.FactorizedTensor.new(self.factorization_shape,
                                                   rank=rank, factorization="Tucker",
                                                   device=device, dtype=dtype)
        if(bias):
            self.bias = torch.nn.parameter.Parameter(torch.zeros((out_channels), device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups


    @torch.no_grad()
    def reset_parameters(self, std = 1, std_bias = 1, generator=None):
        '''
        Sets the parameters so that the weights (and bias) will be normal of mean 0 with the given standard deviation.
        The factorized weights are initialized using tltorch.factorized_tensors.init.tucker_init to acheive this.
        '''
        if(self.bias is not None):
            torch.nn.init.normal_(self.bias, mean=0.0, std=std_bias, generator=generator)

        tltorch.factorized_tensors.init.tucker_init(self.weight_tucker, std=std)

    @property
    def weight(self) -> torch.Tensor:
        '''
        The full, unfactorized convolution weights
        '''
        return self.weight_tucker.to_tensor().view(self.weights_shape)

    def forward(self, X):

        return SpatioTemporalTuckerRFConv3D.tucker_conv(X,
                                                        self.weight_tucker,
                                                        self.factorization_type,
                                                        self.weights_shape,
                                                        bias=self.bias,
                                                        stride=self.stride,
                                                        padding=self.padding,
                                                        dilation=self.dilation)

    @staticmethod
    def tucker_conv(x : torch.tensor,
                    tucker_tensor : tltorch.TuckerTensor,
                    factorization_type : str,
                    weight_shape_0 : tuple, 
                    bias : torch.Tensor = None, 
                    stride : int | Tuple[int] = 1,
                    padding : int | Tuple[int] = 0,
                    dilation : int | Tuple[int] = 1):
        '''
        Specialized version slightly altered from the tensory torch library for my space-time separable kernels for reshaping space
        tltorch.functional.convolutions.tucker_conv
        '''
        # Extract the rank from the actual decomposition in case it was changed by, e.g. dropout
        rank =tucker_tensor.rank

        weight_shape = list(weight_shape_0)
        weight_shape[0] = rank[0]

        batch_size = x.shape[0]
        n_dim = tl.ndim(x)

        # Change the number of channels to the rank
        x_shape = list(x.shape)

        # only do if not combining space & in channel factors
        if(factorization_type == 'spatial'):
            x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

            # This can be done with a tensor contraction
            # First conv == tensor contraction
            # from (in_channels, rank) to (rank == out_channels, in_channels, 1)
            ker = tl.transpose(tucker_tensor.factors[1]).unsqueeze(2);
            x   = torch.nn.functional.conv1d(x, ker)

            x_shape[1] = rank[1]
            x = x.reshape(x_shape)

            start_factors_mult = 2
            weight_shape[1] = rank[1]
        else:
            start_factors_mult = 1

        modes  = list(range(start_factors_mult, n_dim+1))
        weight = tl.tenalg.multi_mode_dot(tucker_tensor.core, tucker_tensor.factors[start_factors_mult:], modes=modes)
        
        #reshape weight to the correct x-y space
        weight = weight.reshape(weight_shape)

        # print(f"weight shape {weight.shape}")
        # print(f"x shape {x.shape}")


        x = tltorch_convolve(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation)

        # Revert back number of channels from rank to output_channels
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1))
        # Last conv == tensor contraction
        # From (out_channels, rank) to (out_channels, in_channels == rank, 1)
        x = torch.nn.functional.conv1d(x, tucker_tensor.factors[0].unsqueeze(2), bias=bias)

        x_shape[1] = x.shape[1]
        x = x.reshape(x_shape)

        return x
 
    def __repr__(self):
        msg = (f'{self.__class__.__name__} wrapping a tensorly torch factorized Conv3D('
               + f'{self.in_channels}, {self.out_channels},'
               + f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias is not None})')
        msg += f', with full weight size {self.weights_shape} with factorization_type={self.factorization_type} as a {self.weight_tucker.__repr__()}'
        return msg

class BatchNorm3DSpace(torch.nn.BatchNorm2d):
    '''
    BatchNorm2D for inputs of size (N,C,X,Y,T) where we want to normalize over each X,Y but not T.

    Forward simply rearranged input to be (N,C*T,X,Y), calls torch.nn.BatchNorm2D, and then rearranges things back.
    Besides the size time_bins (T), the options are passed to the torch.nn.BatchNorm2D constructor.
    '''
    def __init__(self, in_channels : int, time_bins : int, **kwargs):
        super().__init__(in_channels*time_bins, **kwargs)

    def forward(self, X):
        X = torch.movedim(X,4,2)
        shape_0 = X.shape
        X = X.view(-1, self.num_features, X.shape[-2], X.shape[-1])

        X = torch.nn.BatchNorm2d.forward(self,X)

        X = X.view(shape_0)
        X = torch.movedim(X,2,4)

        return X


class InstanceNorm3DSpace(torch.nn.InstanceNorm2d):
    '''
    InstanceNorm2d for inputs of size (N,C,X,Y,T) where we want to normalize over each X,Y but not T.

    Forward simply rearranged input to be (N,C*T,X,Y), calls torch.nn.InstanceNorm2d, and then rearranges things back.
    Besides the size time_bins (T), the options are passed to the torch.nn.InstanceNorm2d constructor.
    '''
    def __init__(self, in_channels : int, time_bins : int, **kwargs):
        super().__init__(in_channels*time_bins, **kwargs)

    def forward(self, X):
        X = torch.movedim(X,4,2)
        shape_0 = X.shape
        X = X.view(-1, self.num_features, X.shape[-2], X.shape[-1])

        X = torch.nn.InstanceNorm2d.forward(self,X)

        X = X.view(shape_0)
        X = torch.movedim(X,2,4)

        return X


'''
===============================================================================
===============================================================================
1-D conv net model for neural response time series with full-field stimuli
===============================================================================
===============================================================================
'''

class PopulationFullFieldNet(torch.nn.Sequential):
    # two layer, 1-D convolution followed by fullly connected layer
    def __init__(self, num_cells : int,
                       in_channels : int = 1,
                       layer_channels : int | Tuple[int] = (8,8),
                       layer_time_lengths : int | Tuple[int] = (40,12),
                       layer_groups : int | Tuple[int] = 1,
                       layer_bias : bool | Tuple[bool] = True,
                       layer_nonlinearity : bool | Tuple[bool] = True,
                       hidden_activation : Optional[Callable[..., torch.nn.Module]] = torch.nn.Softplus,
                       out_activation    : Optional[Callable[..., torch.nn.Module]] = torch.nn.Softplus, #None, torch.nn.Softplus,
                       out_bias : bool = True,
                       device=None, dtype=None, verbose : bool = True):
        '''
        Builds a sequential model as a series of temporal convolution (1-D) layers, followed by a fully connected linear layer for the output.
        This model does not include normalization layers right now.
        Args:
            num_cells : the number of neurons in the population being modelled (output size)
            in_channels : (C) the number of input channels
            The following are lists for each convolution layer (or a constant if the same for each layer, except for layer_channels)
                layer_channels : the number of output channels. 
                                 IMPORTANT: The length of this parameter controls the number of layers. If it is a constant, only 1 layer will be created
                layer_time_lengths : convolution kernel time length
                layer_groups : the convolution groups parameter.
                layer_bias : to include bias term
                layer_nonlinearity : true/false if a nonlinearity is used after this layer.
            hidden_activation : the activation function to be used after each convolution layer (if layer_nonlinearity is True)
            out_bias : if final linear layer has a bias
            out_activation : the activation function following the final output nonlinearity
            verbose : to print model info during instantiation
            device : which device to place model on
            dtype : the dtype of all the layers

            Expected input-output size for N samples of length T: (N,C,T+time_padding) -> (N,num_cells,T)
        '''
        
        layer_channels = tuple_convert(layer_channels)
        n_layers = len(layer_channels);

        in_channels = 1
        layer_time_lengths = tuple_convert(layer_time_lengths, n_layers)
        layer_groups = tuple_convert(layer_groups, n_layers)
        layer_nonlinearity = tuple_convert(layer_nonlinearity, n_layers)
        layer_bias = tuple_convert(layer_bias, n_layers)
        
        assert len(layer_channels) == len(layer_time_lengths), "must give same number of layer_time_lengths and layer_channels"
        assert len(layer_channels) == len(layer_groups), "must give same number of layer_groups and layer_channels"
        assert len(layer_channels) == len(layer_nonlinearity), "must give same number of layer_nonlinearity and layer_channels"
        assert len(layer_channels) == len(layer_bias), "must give same number of layer_bias and layer_channels"

        if(verbose): print(f"Building multi-layer temporal convolutional model for {num_cells} neurons and full-field stimuli.")

        layers = []
        for ker_size, out_channels, groups_c, nonlinearity_c, bias_c in zip(layer_time_lengths,
                                                    layer_channels,
                                                    layer_groups,
                                                    layer_nonlinearity,
                                                    layer_bias):
            # linear layer
            if(verbose): print(f"Adding full-rank convolutional layer of input length {ker_size} and {out_channels} channels.")
            layers.append(torch.nn.Conv1d(in_channels=in_channels,
                                          out_channels=out_channels, groups=groups_c,
                                          kernel_size=ker_size,
                                          stride=1, bias=bias_c, padding="valid",
                                          device=device, dtype=dtype))
            in_channels = out_channels
            # add activation
            if(hidden_activation is not None and nonlinearity_c):
                if(verbose): print(f"Adding nonlinearity: {hidden_activation.__name__}.")
                layers.append(hidden_activation())

        # output layer: fully connected
        if(verbose): print(f"Adding full-connected linear layer: {in_channels} to {num_cells}.")
        layers.append(TransposeLinear(in_features=in_channels,
                                      out_features=num_cells,
                                      bias=out_bias,
                                      device=device, dtype=dtype
                                      ))
        
        if(not out_activation is None):
            if(verbose): print(f"Adding output nonlinearity: {out_activation.__name__}.")
            layers.append(out_activation())
            self.nonlinear_output = True
        else:
            self.nonlinear_output = False
        
        super().__init__(*layers)

        self.layer_time_lengths_ = layer_time_lengths
        self.layer_channels_ = layer_channels
        self.num_cells_ = num_cells

        if(verbose): print(f"Model initialized.")

    @property
    def time_padding(self) -> int:
        '''
        The total number of time bins that are needed to be given in each input preceding the first output.
        input-output size: (N,C,T+time_padding) -> (N,num_cells,T)
        '''
        return np.sum(self.layer_time_lengths_)-len(self.layer_time_lengths_)

'''
===============================================================================
===============================================================================
3-D conv net model for neural response time series with video stimuli
===============================================================================
===============================================================================
'''

class PopulationConvNet(torch.nn.Sequential):
    # 3-D convolution layers followed by one fully connected layer (for individual neurons)
    def __init__(self, num_cells : int,
                       frame_width : int ,
                       frame_height : int,
                       in_channels : int = 1,
                       in_time : int = 1,
                       layer_channels : int | Tuple[int] = (8,8),
                       layer_time_lengths : int | Tuple[int] = (40,12),
                       layer_spatio_temporal_rank : int | List[int|Tuple] = 2,
                       layer_spatio_temporal_factorization_type : str | Tuple[str] = 'spatial',

                       layer_rf_pixel_widths : int | Tuple[int] = (15,15),
                       layer_rf_pixel_heights : int | Tuple[int] = None,
                       layer_rf_dilations_x : int | Tuple[int] = None,
                       layer_rf_dilations_y : int | Tuple[int] = None,
                       layer_rf_strides_x : int | Tuple[int] = None,
                       layer_rf_strides_y : int | Tuple[int] = None,
                       layer_groups : int | Tuple[int] = 1,
                       layer_bias : bool | Tuple[bool] = True,
                       layer_nonlinearity : bool | Tuple[bool] = True,
                       layer_normalization : bool | str | Tuple = True,
                       hidden_activation : Optional[Callable[..., torch.nn.Module]] = torch.nn.Softplus,
                       out_activation    : Optional[Callable[..., torch.nn.Module]] = torch.nn.Softplus,
                       out_bias : bool = True,
                       out_normalization : bool | str = True,
                       device=None, dtype=None, verbose : bool = True,
                       normalization_options ={"affine" : False,
                                               "track_running_state" : False}
                    ):
        '''
        Builds a sequential model as a series of spatiotemporal convolution (3-D) layers, followed by a fully connected linear layer for the output.
        The model can include normalization layers at each stage.
        Args:
            num_cells : the number of neurons in the population being modelled (output size)
            frame_width : the width of the input video
            frame_height : the height of the input video
            in_channels : (C) the number of input channels
            in_time : (T) the number of time bins per input. Only needed for normalization layers when normalization_options affine=True or track_running_state=True
            The following are lists for each convolution layer (or a constant if the same for each layer, except for layer_channels)
                layer_channels : the number of output channels. 
                                 IMPORTANT: The length of this parameter controls the number of layers. If it is a constant, only 1 layer will be created
                layer_rf_pixel_widths : convultion kernel space width
                layer_rf_pixel_heights : convultion kernel space height
                layer_rf_dilations_x/y : the dilation of the convolution in x/y (note: no dialtion is allowed in time right now)
                layer_rf_strides_x/y : the stride of the convolution in x/y (note: stride in time is fixed to 1)
                layer_time_lengths : convolution kernel time length
                layer_groups : the convolution groups parameter.
                layer_bias : to include bias term
                layer_spatio_temporal_factorization_type : 'spatial'  'spatialchannel', or None. How to factorize the weights for the convolution using a Tucker decomposition.
                                                           If None, uses a full-rank convolution (standard torch.nn.Conv3D).   
                layer_spatio_temporal_rank : the rank of the Tucker decomposition of the weights if layer_spatio_temporal_factorization_type is not None                                                                                                 
                layer_nonlinearity : true/false if a nonlinearity is used after this layer
                layer_normalization : to include a normalization in the input to this channel. I
                                      If True, then BatchNorm2D over space.
                                      If '3D', then BatchNorm3D over space/time.
                                      If 'instance', then InstanceNorm2D over space.
                                      If 'instance3D', then InstanceNorm3D over space/time.
                
            hidden_activation : the activation function to be used after each convolution layer (if layer_nonlinearity is True)
            out_bias : if final linear layer has a bias
            out_activation : the activation function following the final output nonlinearity
            out_normalization : to include a normalization or not before the final output linearity 
            normalization_options : keyword arguments to normalization layers. Right now, the same args are passed to all layers.
                                    By default, no momentum or affine weight is used. This choice makes sizing inputs easier. 
            verbose : to print model info during instantiation
            device : which device to place model on
            dtype : the dtype of all the layers

            Expected input-output size for N samples of length T: (N,C,frame_width,frame_height,T+time_padding) -> (N,num_cells,T)
        '''

        if(in_channels is None):
            in_channels = 1

        layer_channels = tuple_convert(layer_channels)
        n_layers = len(layer_channels);

        layer_time_lengths = tuple_convert(layer_time_lengths, n_layers)
        layer_rf_pixel_widths = tuple_convert(layer_rf_pixel_widths, n_layers)
        layer_rf_pixel_heights = tuple_convert(layer_rf_pixel_heights, n_layers, layer_rf_pixel_widths)

        layer_rf_dilations_x = tuple_convert(layer_rf_dilations_x, n_layers)
        layer_rf_dilations_y = tuple_convert(layer_rf_dilations_y, n_layers, layer_rf_dilations_y)
        layer_rf_strides_x = tuple_convert(layer_rf_strides_x, n_layers)
        layer_rf_strides_y = tuple_convert(layer_rf_strides_y, n_layers, layer_rf_strides_x)

        if(isinstance(layer_spatio_temporal_rank, int) or isinstance(layer_spatio_temporal_rank, tuple)):
            layer_spatio_temporal_rank = [layer_spatio_temporal_rank] * n_layers
        layer_spatio_temporal_factorization_type = tuple_convert(layer_spatio_temporal_factorization_type, n_layers)
        # layer_spatio_temporal_rank = tuple_convert(layer_spatio_temporal_rank, n_layers)

        layer_bias = tuple_convert(layer_bias, n_layers)
        layer_groups = tuple_convert(layer_groups, n_layers)
        layer_nonlinearity = tuple_convert(layer_nonlinearity, n_layers)
        layer_normalization = tuple_convert(layer_normalization, n_layers)
        
        assert len(layer_channels) == len(layer_time_lengths), "must give same number of layer_time_lengths and layer_channels"
        assert len(layer_channels) == len(layer_rf_pixel_widths), "must give same number of layer_rf_pixel_widths and layer_channels"
        assert len(layer_channels) == len(layer_rf_pixel_heights), "must give same number of layer_rf_pixel_heights and layer_channels"
        assert len(layer_channels) == len(layer_rf_dilations_x), "must give same number of layer_rf_dilations_x and layer_channels"
        assert len(layer_channels) == len(layer_rf_dilations_y), "must give same number of layer_rf_dilations_y and layer_channels"
        assert len(layer_channels) == len(layer_rf_strides_x), "must give same number of layer_rf_strides_x and layer_channels"
        assert len(layer_channels) == len(layer_rf_strides_y), "must give same number of layer_rf_strides_y and layer_channels"
        assert len(layer_channels) == len(layer_spatio_temporal_rank), "must give same number of layer_spatio_temporal_rank and layer_channels"
        assert len(layer_channels) == len(layer_groups), "must give same number of layer_groups and layer_channels"
        assert len(layer_channels) == len(layer_nonlinearity), "must give same number of layer_nonlinearity and layer_channels"
        assert len(layer_channels) == len(layer_bias), "must give same number of layer_bias and layer_channels"
        assert len(layer_channels) == len(layer_normalization), "must give same number of layer_bias and layer_normalization"

        layers = []
        

        width = frame_width
        height = frame_height
        total_time_padding = 0

        if(verbose): print(f"Building multi-layer convolutional model for {num_cells} neurons and image size {width} x {height}")

        for (ker_time, ker_width, ker_height, 
             out_channels, 
             stride_x, stride_y, 
             dilation_x, dilation_y, 
             st_rank, factorization_type, 
             groups_c, 
             nonlinearity_c, 
             normalization_c,
             bias_c
            ) in zip(layer_time_lengths,
                    layer_rf_pixel_widths,
                    layer_rf_pixel_heights,
                    layer_channels,
                    layer_rf_strides_x,
                    layer_rf_strides_y,
                    layer_rf_dilations_x,
                    layer_rf_dilations_y,
                    layer_spatio_temporal_rank,
                    layer_spatio_temporal_factorization_type,
                    layer_groups,
                    layer_nonlinearity,
                    layer_normalization,
                    bias_c):
            

            padding_x = 0;
            padding_y = 0;
            padding_t = 0;

            kernel_size = (ker_width, ker_height, ker_time)

            
            input_time_bins_c = max(1,input_time_bins_c - total_time_padding)

            if(normalization_c == True):
                if(verbose): print(f"Adding 2D batch normalization layer.")
                layers.append(BatchNorm3DSpace(in_channels=in_channels, time_bins=input_time_bins_c, **normalization_options))
            elif(normalization_c == 'instance'):
                if(verbose): print(f"Adding 2D instance normalization layer.")
                layers.append(InstanceNorm3DSpace(in_channels=in_channels, time_bins=input_time_bins_c, **normalization_options))
            elif(normalization_c == '3D'):
                if(verbose): print(f"Adding 3D batch normalization layer.")
                layers.append(torch.nn.BatchNorm3d(in_channels=in_channels, **normalization_options))
            elif(normalization_c == 'instance3D'):
                if(verbose): print(f"Adding 3D instance normalization layer.")
                layers.append(torch.nn.InstanceNorm3d(in_channels=in_channels, **normalization_options))
            else:
                raise ValueError(f"Unkown normalization option: {normalization_c}")


            if(layer_spatio_temporal_factorization_type is None):
                if(verbose): print(f"Adding full-rank convolutional layer of size {kernel_size} and {out_channels} channels.")
                layers.append(torch.nn.Conv3d(in_channels=in_channels,
                                            out_channels=out_channels, groups=groups_c,
                                            kernel_size=kernel_size,
                                            stride=(stride_x, stride_y, 1), dilation=(dilation_x, dilation_y, 1),
                                            bias=bias_c, padding=(padding_x, padding_y, padding_t),
                                            device=device, dtype=dtype))
            else:
                if(verbose): print(f"Adding Tucker convolutional layer of size {kernel_size} and {out_channels} channels with factorization type {factorization_type} and rank {st_rank}.")
                layers.append(SpatioTemporalTuckerRFConv3D(in_channels=in_channels,
                                            out_channels=out_channels, groups=groups_c,
                                            kernel_size=kernel_size,
                                            rank=st_rank,
                                            factorization_type = factorization_type,
                                            stride=(stride_x, stride_y, 1), dilation=(dilation_x, dilation_y, 1),
                                            bias=bias_c, padding=(padding_x, padding_y, padding_t),
                                            device=device, dtype=dtype))


            in_channels = out_channels

            height = compute_conv_output_size(height, ker_height, stride_y, dilation_y, padding_y)
            width  = compute_conv_output_size(width,  ker_width, stride_x, dilation_x, padding_x)
            total_time_padding += ker_time-1

            if( hidden_activation is not None and nonlinearity_c):
                if(verbose): print(f"Adding nonlinearity: {hidden_activation.__name__}.")
                layers.append(hidden_activation())

        # output layer: fully connected
        
        input_time_bins_c = max(1,input_time_bins_c - total_time_padding)

        if(out_normalization == True):
            if(verbose): print(f"Adding final 2D batch normalization layer.")
            layers.append(BatchNorm3DSpace(in_channels=in_channels, time_bins=input_time_bins_c, **normalization_options))
        elif(out_normalization == 'instance'):
            if(verbose): print(f"Adding final 2D instance normalization layer.")
            layers.append(InstanceNorm3DSpace(in_channels=in_channels, time_bins=input_time_bins_c, **normalization_options))
        elif(out_normalization == '3D'):
            if(verbose): print(f"Adding final 3D batch normalization layer.")
            layers.append(torch.nn.BatchNorm3d(in_channels=in_channels, **normalization_options))
        elif(out_normalization == 'instance3D'):
            if(verbose): print(f"Adding final 3D instance normalization layer.")
            layers.append(torch.nn.InstanceNorm3d(in_channels=in_channels, **normalization_options))
        else:
            raise ValueError(f"Unknown normalization option: {out_normalization}")
            
        in_features = [in_channels, width, height]
        if(verbose): print(f"Adding full-connected linear layer: {in_features} to {num_cells}.")
        layers.append(TransposeLinear(in_features=in_features,
                                      out_features=num_cells,
                                      bias=out_bias,
                                      device=device, dtype=dtype
                                      ))
        if(not out_activation is None):
            if(verbose): print(f"Adding output nonlinearity: {out_activation.__name__}.")
            layers.append(out_activation())
            self.nonlinear_output = True
        else:
            self.nonlinear_output = False
        
        super().__init__(*layers)

        self.layer_time_lengths_ = layer_time_lengths
        self.layer_channels_ = layer_channels
        self.num_cells_ = num_cells

        self.frame_width_ = frame_width
        self.frame_height_ = frame_height

        if(verbose): print(f"Model initialized.")

    @property
    def time_padding(self) -> int:
        '''
        The total number of time bins that are needed to be given in each input preceding the first output.
        input-output size: (N,C,X,Y,T+time_padding) -> (N,num_cells,T)
        '''
        return np.sum(self.layer_time_lengths_)-len(self.layer_time_lengths_)

'''
===============================================================================
===============================================================================
Regularization functions
- Elastic net + fused lasso penalties (in space&time dimensions) on convolution kernels
===============================================================================
===============================================================================
'''    

def conv_penalty(K : torch.Tensor, en_lambda = None, en_alpha : float = None, fl_lambda_x = None, fl_lambda_y = None, fl_lambda_t : float = None) -> torch.Tensor:
    '''
    Sum of elastic net & fused lasso penalties
    Args:
        K : (..., [space_x, space_y], time) : The kernel to penalize
            Space dimensions only needed if spatiotemporal kernels and fl_lambda_x/y are used.
        en_lambda : [0,inf) the elastic net weight
        en_alpha  : [0,1] the elastic net weight between l1 or l2 penalties. 
                    e.g.: en_alpha = 1 means l2 penalty is 0 and l1 penalty is en_lambda.
                          en_alpha = 0.25 means l2 penalty is 0.75*en_lambda and l1 penalty is 0.25*en_lambda.
        fl_lambda_x/y/z : weights for the fused lasso penalty along each dimension
    Returns:
        A scalar total penalty
    '''

    l2_p = en_lambda*(1-en_alpha)/2
    l1_p = en_lambda*en_alpha

    if(l2_p is not None and l2_p > 0):
        D = torch.square(K)
        sl2 = l2_p*torch.mean(D)
    else:
        sl2 = 0

    if(l1_p is not None and l1_p > 0):
        D = torch.square(K)
        sl1 = l1_p*torch.mean(D)
    else:
        sl1 = 0

    if(fl_lambda_t is not None and fl_lambda_t > 0  and K.shape[-1] > 1):
        D = torch.diff(K,dim=-1)
        D = torch.abs(D)
        st = torch.mean(D) * fl_lambda_t
    else:
        st = 0

    if(fl_lambda_y is not None and fl_lambda_y > 0 and K.shape[-2] > 1):
        D = torch.diff(K,dim=-2)
        D = torch.abs(D)
        sy = torch.mean(D) * fl_lambda_y
    else:
        sy = 0

    if(fl_lambda_x is not None and fl_lambda_x > 0 and K.shape[-3] > 1):
        D = torch.diff(K,dim=-3)
        D = torch.abs(D)
        sx = torch.mean(D) * fl_lambda_x
    else:
        sx = 0

    return sx + sy + st + sl2 + sl1

def penalize_convnet_weights(model : torch.nn.Sequential,
                             en_lambda : float | Tuple[float] = None,
                             en_alpha : float | Tuple[float]  = None,
                             fl_lambda_x : float | Tuple[float]  = None,
                             fl_lambda_y  : float | Tuple[float] = None,
                             fl_lambda_t : float | Tuple[float]  = None,
                             lin_en_lambda  : float | Tuple[float] = None,
                             lin_en_alpha  : float | Tuple[float] = None) -> torch.Tensor:
    '''
    Computes the sum of elastic net & fused lasso penalties on the weights of a sequence of linear layers.
    For fully connected layers, only the elastic net is used.
    Args:
        model : the sequential network model. Ideally is one of the models defined in this module
        en_lambda : [0,inf) the elastic net weight. Can be scalar or tuple containing penalty for each linear layer (in order).
        en_alpha  : [0,1] the elastic net weight between l1 or l2 penalties.  Can be scalar or tuple containing penalty for each linear layer (in order).
                    e.g.: en_alpha = 1 means l2 penalty is 0 and l1 penalty is en_lambda.
                          en_alpha = 0.25 means l2 penalty is 0.75*en_lambda and l1 penalty is 0.25*en_lambda.
        fl_lambda_x/y/z : weights for the fused lasso penalty along each dimension. Can be scalar or tuple containing penalty for each linear layer (in order).
    Returns:
        A scalar total penalty to be added to a loss function.
    '''
    ctr = 0
    lin_ctr = 0

    num_lin_layers = 0
    for xx in model:
        if(isinstance(xx, tltorch.FactorizedConv | torch.nn.Conv1d | torch.nn.Conv2d | torch.nn.Conv3d | SpatioTemporalTuckerRFConv3D | torch.nn.Linear | TransposeLinear)):
            num_lin_layers += 1

    base = (None,) * num_lin_layers

    en_lambda = tuple_convert(en_lambda, num_lin_layers, base)
    en_alpha  = tuple_convert(en_alpha,  num_lin_layers, base)
    fl_lambda_x  = tuple_convert(fl_lambda_x, num_lin_layers, base)
    fl_lambda_y  = tuple_convert(fl_lambda_y, num_lin_layers, base)
    fl_lambda_t  = tuple_convert(fl_lambda_t, num_lin_layers, base)

    lin_en_lambda = tuple_convert(lin_en_lambda, num_lin_layers, base)
    lin_en_alpha  = tuple_convert(lin_en_alpha, num_lin_layers, base)

    p = 0
    for xx in model:
        if(isinstance(xx, torch.nn.Conv1d)):
            p += conv_penalty(xx.weight, en_lambda=en_lambda[ctr], en_alpha=en_alpha[ctr], fl_lambda_t=fl_lambda_t[ctr])
            ctr += 1
        elif(isinstance(xx, torch.nn.Conv3d | SpatioTemporalTuckerRFConv3D)):
            p += conv_penalty(xx.weight, en_lambda=en_lambda[ctr], en_alpha=en_alpha[ctr], fx_lambda_t=fl_lambda_x[ctr], fl_lambda_y=fl_lambda_y[ctr], fl_lambda_t=fl_lambda_t[ctr])
            ctr += 1
        elif(isinstance(xx, torch.nn.Linear | TransposeLinear)):
            p += conv_penalty(xx.weight, en_lambda=lin_en_lambda[lin_ctr], en_alpha=lin_en_alpha[lin_ctr])
            lin_ctr += 1
        elif(isinstance(xx, torch.nn.Conv2d)):
            raise NotImplementedError("Not implemented for 2D convs yet: not sure if 2D is space/time or space/space.")
        elif(isinstance(xx, tltorch.FactorizedConv)):
            raise NotImplementedError("Not implemented for tltorch factorized convultions yet.")
    return p