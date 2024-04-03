import torch
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import numpy.typing as npt
from itertools import cycle
# import tltorch

def compute_conv_output_size(L_in : int, kernel_size : int, stride : int = 1,
                             dilation : int=1, padding : int = 0) -> int:
    L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
    return L_out

def tuple_convert(a, l = 1, target = None):
    if(a is None):
        return target
    elif(isinstance(a, list)):
        return tuple(a)
    elif(isinstance(a, tuple)):
        return a
    else:
        try:
            if(isinstance(a,npt.ArrayLike)):
                return tuple(a)
        except:
            pass
        return (a,) * l
        

            

class SpatioTemporalRFConv3D(torch.nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : tuple, rank : int, bias : bool = True, stride=1, padding=0, dilation=1, groups = 1):
        
        super().__init__()

        kernel_size = tuple_convert(kernel_size)
        assert len(kernel_size) == 3, "kernel size must be length 3"
        assert rank > 0 and rank < (kernel_size[0] * kernel_size[1] * (in_channels/groups)) and rank < kernel_size[2], "rank is not within valid limits"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank

        self.space_channels  = int(in_channels/groups) * kernel_size[0] * kernel_size[1]

        self.weights_shape_ = (out_channels, int(in_channels/groups)) +  kernel_size

        # self.weights_space_ = torch.zeros(out_channels, in_channels/groups, kernel_size[0], kernel_size[1], rank)
        self.weights_space_ = torch.nn.parameter.Parameter(torch.zeros((out_channels, self.space_channels, rank)))
        self.weights_time_  = torch.nn.parameter.Parameter(torch.zeros((out_channels, rank, kernel_size[2])))
        if(bias):
            self.bias = torch.nn.parameter.Parameter(torch.zeros((out_channels)))
        else:
            self.bias = None
        self.init_weights()

        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups


    @torch.no_grad()
    def init_weights(self):
        if(self.bias is not None):
            self.bias[:] = torch.randn(self.out_channels)

        weights_full = torch.randn(self.out_channels, self.space_channels, self.kernel_size[2]) / np.sqrt(self.space_channels)
        self.refactorize_weights(weights_full)

    @torch.no_grad()
    def refactorize_weights(self, W = None):
        if(W is None):
            W = torch.matmul(self.weights_space_, self.weights_time_ )
        U,S,Vh = torch.linalg.svd(W)
        
        U = U[:,:,:self.rank]
        Vh = Vh[:,:self.rank,:]
        S = torch.unsqueeze(S[:,:self.rank], -1)
        self.weights_space_[:,:,:] = U
        self.weights_time_[:,:,:]  = Vh * S

    def weight(self):
        W = torch.matmul(self.weights_space_, self.weights_time_ )
        W = W.view(self.weights_shape_)
        return W

    
    def forward(self, X):
        W = self.weight()

        return torch.nn.functional.conv3d(X, W, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

# 
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
        self.in_features_total = np.sum(in_features)
        self.n_in_features = len(in_features)

        self.linear_ = torch.nn.Linear(self.in_features_total, out_features, **kwargs)

    def forward(self, X):
        if(self.n_in_features > 1):
            # for flattening results of a Conv3D conv into the right fully connected layer
            X = torch.nn.Flatten(start_dim=-1-self.n_in_features, end_dim=-2);
        
        # flips dimensions
        X = torch.transpose(X, dim0=-2, dim1=-1)
        # linear op
        X = self.linear_(X)
        # flips dimensions back: idea is to get a (N_samples, Neuron, Time) output tensor
        X = torch.transpose(X, dim0=-2, dim1=-1)
        return X


# specific convolutional neural nets for fitting RFs of visual neurons

class PopulationFullFieldNet(torch.nn.Sequential):
    # two layer, 1-D convolution followed by fullly connected layer
    def __init__(self, num_cells : int,
                       layer_channels : int | Tuple[int] = [8,8],
                       layer_time_lengths : int | Tuple[int] = [100,40],
                       hidden_activation : Optional[Callable[..., torch.nn.Module]] = torch.nn.Softplus,
                       out_activation    : Optional[Callable[..., torch.nn.Module]] = None, #torch.nn.Softplus,
                       bias : bool = True):
        

        layer_channels = tuple_convert(layer_channels)
        n_layers = len(layer_channels);

        in_channels = 1
        layer_time_lengths = tuple_convert(layer_time_lengths, n_layers)
        
        assert len(layer_channels) == len(layer_time_lengths), "must give same number of layer_time_lengths and layer_channels"

        layers = []
        in_channels = 1
        for ker_size, out_channels in zip(layer_time_lengths, layer_channels):

            layers.append(torch.nn.Conv1d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=ker_size,
                                          stride=1, bias=bias, padding="valid"))
            in_channels = out_channels

            if(not hidden_activation is None):
                layers.append(hidden_activation())

        # output layer: fully connected
        layers.append(TransposeLinear(in_features=in_channels,
                                      out_features=num_cells,
                                      bias=bias
                                      ))
        
        if(not out_activation is None):
            layers.append(out_activation())
        
        super().__init__(*layers)

        self.layer_time_lengths_ = layer_time_lengths
        self.layer_channels_ = layer_channels
        self.num_cells_ = num_cells

    @property
    def time_padding(self) -> int:
        return np.sum(self.layer_time_lengths_)-len(self.layer_time_lengths_)
    
class PopulationConvNet(torch.nn.Sequential):
    # 3-D convolution layers followed by one fully connected layer (for individual neurons)
    def __init__(self, num_cells : int,
                       frame_width : int ,
                       frame_height : int,
                       layer_channels : int | Tuple[int] = (8,8),
                       layer_time_lengths : int | Tuple[int] = (100,40),
                       layer_spatio_temporal_rank : int | Tuple[int] = 2,

                       layer_rf_pixel_widths : int | Tuple[int] = (15,15),
                       layer_rf_pixel_heights : int | Tuple[int] = None,
                       layer_rf_dilations_x : int | Tuple[int] = None,
                       layer_rf_dilations_y : int | Tuple[int] = None,
                       layer_rf_strides_x : int | Tuple[int] = None,
                       layer_rf_strides_y : int | Tuple[int] = None,
                       hidden_activation : Optional[Callable[..., torch.nn.Module]] = torch.nn.Softplus,
                       out_activation    : Optional[Callable[..., torch.nn.Module]] = None,#torch.nn.Softplus,
                       bias : bool = True):
        

        layer_channels = tuple_convert(layer_channels)
        n_layers = len(layer_channels);

        layer_time_lengths = tuple_convert(layer_time_lengths, n_layers)
        layer_rf_pixel_widths = tuple_convert(layer_rf_pixel_widths, n_layers)
        layer_rf_pixel_heights = tuple_convert(layer_rf_pixel_heights, n_layers, layer_rf_pixel_widths)

        layer_rf_dilations_x = tuple_convert(layer_rf_dilations_x, n_layers)
        layer_rf_dilations_y = tuple_convert(layer_rf_dilations_y, n_layers, layer_rf_dilations_y)
        layer_rf_strides_x = tuple_convert(layer_rf_strides_x, n_layers)
        layer_rf_strides_y = tuple_convert(layer_rf_strides_y, n_layers, layer_rf_strides_x)

        layer_spatio_temporal_rank = tuple_convert(layer_spatio_temporal_rank, n_layers)
        
        assert len(layer_channels) == len(layer_time_lengths), "must give same number of layer_time_lengths and layer_channels"
        assert len(layer_channels) == len(layer_rf_pixel_widths), "must give same number of layer_rf_pixel_widths and layer_channels"
        assert len(layer_channels) == len(layer_rf_pixel_heights), "must give same number of layer_rf_pixel_heights and layer_channels"
        assert len(layer_channels) == len(layer_rf_dilations_x), "must give same number of layer_rf_dilations_x and layer_channels"
        assert len(layer_channels) == len(layer_rf_dilations_y), "must give same number of layer_rf_dilations_y and layer_channels"
        assert len(layer_channels) == len(layer_rf_strides_x), "must give same number of layer_rf_strides_x and layer_channels"
        assert len(layer_channels) == len(layer_rf_strides_y), "must give same number of layer_rf_strides_y and layer_channels"
        assert len(layer_channels) == len(layer_spatio_temporal_rank), "must give same number of layer_spatio_temporal_rank and layer_channels"

        layers = []
        in_channels = 1

        width = frame_width
        height = frame_height
        for ker_time, ker_width, ker_height, out_channels, stride_x, stride_y, dilation_x, dilation_y, st_rank in zip(layer_time_lengths,
                    layer_rf_pixel_widths,
                    layer_rf_pixel_heights,
                    layer_channels,
                    layer_rf_strides_x,
                    layer_rf_strides_y,
                    layer_rf_dilations_x,
                    layer_rf_dilations_y,
                    layer_spatio_temporal_rank):

            padding_x = 0;
            padding_y = 0;
            padding_t = 0;

            if(st_rank <= 0 or st_rank >= np.min(ker_height*ker_width, ker_time)):
                layers.append(torch.nn.Conv3d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(ker_width, ker_height, ker_time),
                                            stride=(stride_x, stride_y, 1), dilation=(dilation_x, dilation_y, 1), bias=bias, padding=(padding_x, padding_y, padding_t)))
            else:
                if(ker_time <= 1):
                    raise NotImplementedError("no time conv")
                else:
                    layers.append(SpatioTemporalRFConv3D(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(ker_width, ker_height, ker_time),
                                            stride=(stride_x, stride_y, 1), dilation=(dilation_x, dilation_y, 1), bias=bias, padding=(padding_x, padding_y, padding_t)))


            in_channels = out_channels

            height = compute_conv_output_size(height, ker_height, stride_y, dilation_y, padding_y)
            width  = compute_conv_output_size(width,  ker_width, stride_x, dilation_x, padding_x)

            if(not hidden_activation is None):
                layers.append(hidden_activation())

        # output layer: fully connected
        layers.append(TransposeLinear(in_features=[in_channels, width, height],
                                      out_features=num_cells,
                                      bias=bias
                                      ))
        if(not out_activation is None):
            layers.append(out_activation())
        
        super().__init__(*layers)

        self.layer_time_lengths_ = layer_time_lengths
        self.layer_channels_ = layer_channels
        self.num_cells_ = num_cells

        self.frame_width_ = frame_width
        self.frame_height_ = frame_height

    @property
    def time_padding(self) -> int:
        return np.sum(self.layer_time_lengths_)-len(self.layer_time_lengths_)
    

def conv_penalty(K, en_lambda = None, en_alpha = None, fl_lambda_x = None, fl_lambda_y = None, fl_lambda_t = None):

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

    if(fl_lambda_t is not None and fl_lambda_t > 0):
        D = torch.diff(K,dim=-1)
        D = torch.abs(D)
        st = torch.mean(D) * fl_lambda_t
    else:
        st = 0

    if(fl_lambda_y is not None and fl_lambda_y > 0):
        D = torch.diff(K,dim=-2)
        D = torch.abs(D)
        sy = torch.mean(D) * fl_lambda_y
    else:
        sy = 0

    if(fl_lambda_x is not None and fl_lambda_x > 0):
        D = torch.diff(K,dim=-3)
        D = torch.abs(D)
        sx = torch.mean(D) * fl_lambda_x
    else:
        sx = 0

    return sx + sy + st + sl2 + sl1

def penalize_convnet_weights(model, en_lambda = None, en_alpha = None, fl_lambda_x = None, fl_lambda_y = None, fl_lambda_t = None,
                                    lin_en_lambda = None, lin_en_alpha = None):
    ctr = 0
    lin_ctr = 0

    en_lambda = tuple_convert(en_lambda, len(model), (None,) * len(model))
    en_alpha  = tuple_convert(en_alpha, len(model), (None,) * len(model))
    fl_lambda_x  = tuple_convert(fl_lambda_x, len(model), (None,) * len(model))
    fl_lambda_y  = tuple_convert(fl_lambda_y, len(model), (None,) * len(model))
    fl_lambda_t  = tuple_convert(fl_lambda_t, len(model), (None,) * len(model))

    lin_en_lambda = tuple_convert(lin_en_lambda, len(model), (None,) * len(model))
    lin_en_alpha  = tuple_convert(lin_en_alpha, len(model), (None,) * len(model))

    p = 0
    for xx in model:
        if(isinstance(xx, torch.nn.Conv1d)):
            p += conv_penalty(xx.weight, en_lambda=en_lambda[ctr], en_alpha=en_alpha[ctr], fl_lambda_t=fl_lambda_t[ctr])
            ctr += 1
        elif(isinstance(xx, torch.nn.Conv2d)):
            raise NotImplementedError("Not sure if 2D is space/time or space/space")
        elif(isinstance(xx, torch.nn.Conv3d)):
            p += conv_penalty(xx.weight, en_lambda=en_lambda[ctr], en_alpha=en_alpha[ctr], fx_lambda_t=fl_lambda_x[ctr], fl_lambda_y=fl_lambda_y[ctr], fl_lambda_t=fl_lambda_t[ctr])
            ctr += 1
        elif(isinstance(xx,PopulationConvNet)):
            p += conv_penalty(xx.weight(), en_lambda=en_lambda[ctr], en_alpha=en_alpha[ctr], fx_lambda_t=fl_lambda_x[ctr], fl_lambda_y=fl_lambda_y[ctr], fl_lambda_t=fl_lambda_t[ctr])
            ctr += 1
        elif(isinstance(xx, torch.nn.Linear)):
            p += conv_penalty(xx.weight, en_lambda=lin_en_lambda[lin_ctr], en_alpha=lin_en_alpha[lin_ctr])
            lin_ctr += 1
    return p