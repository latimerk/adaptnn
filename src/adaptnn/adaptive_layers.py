import torch
from typing import Callable, List, Optional, Sequence, Tuple, Union

class SimpleMLP(torch.nn.Sequential):
    def __init__(self, in_features : int, out_features : int,
                 hidden_features : List[int] | int,
                 hidden_activation : Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 out_activation    : Optional[Callable[..., torch.nn.Module]] = torch.nn.Sigmoid,
                 bias : bool = True, out_bias : bool = True):
        if(isinstance(hidden_features, int)):
            hidden_features = [hidden_features];

        layers = []
        for hf in hidden_features:
            layers.append(torch.nn.Linear(in_features=in_features, out_features=hf, bias=bias))
            if(not hidden_activation is None):
                layers.append(hidden_activation())
            in_features = hf;
        
        layers.append(torch.nn.Linear(in_features=in_features, out_features=out_features, bias=out_bias))
        if(not out_activation is None):
            layers.append(out_activation())
        
        super().__init__(*layers)

class AdaptiveLinear(torch.nn.Module):
    def __init__(self, in_features : int, out_features : int, 
                 adapt_in_features : int, adapt_hidden_features : int | List[int],
                 adapt_hidden_activation : Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU, 
                 adapt_out_activation    : Optional[Callable[..., torch.nn.Module]] = torch.nn.Sigmoid,
                 bias : bool= True, adapt_hidden_bias : bool = True, adapt_out_bias : bool = True, center_B_0 : bool = False):
        super().__init__()
        
        self.B_0 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.B_1 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.W   = SimpleMLP(adapt_in_features, out_features=1, hidden_features=adapt_hidden_features,
                             hidden_activation=adapt_hidden_activation,
                             out_activation=adapt_out_activation,
                             bias = adapt_hidden_bias, out_bias=adapt_out_bias)
        self.center_B_0_ = center_B_0;


    def forward(self, x, adapt_x):
        w_ = self.W(adapt_x)
        if(self.center_B_0_):
            return self.B_0(x) + self.B_1(x)*w_
        else:
            return self.B_0(x)*(1-w_) + self.B_1(x)*w_
        
    def compute_adapt(self, adapt_x):
        with torch.no_grad():
            return self.W(adapt_x)
        
class FixedAdaptiveLinear(torch.nn.Module):
    def __init__(self, in_features : int, out_features : int, 
                 bias : bool= True,  center_B_0 : bool = False):
        super().__init__()
        
        self.B_0 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.B_1 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.center_B_0_ = center_B_0;


    def forward(self, x, w):
        if(self.center_B_0_):
            return self.B_0(x) + self.B_1(x)*w
        else:
            return self.B_0(x)*(1-w) + self.B_1(x)*w
        