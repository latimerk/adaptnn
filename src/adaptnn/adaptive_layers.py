import torch
from typing import Callable, List, Optional, Sequence, Tuple, Union

class SimpleMLP(torch.nn.Sequential):
    def __init__(self, in_features : int, out_features : int,
                 hidden_features : List[int] | int,
                 hidden_activation : Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 out_activation    : Optional[Callable[..., torch.nn.Module]] = None,
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
                 adapt_in_features : int,
                 n_modes : int = 2,
                 bias : bool= True, adapt_bias : bool = True, center_B_0 : bool = True):
        super().__init__()

        if(n_modes < 2):
            raise ValueError(f"n_modes must be int >= 2. Received {n_modes}.")
        
        self.n_modes = n_modes
        self.out_features = out_features
        self.in_features = in_features
        self.adapt_in_features = adapt_in_features
        
        self.B = torch.nn.Linear(in_features=in_features, out_features=out_features*n_modes, bias=bias)

        if(self.n_modes == 2):
            self.center_B_0_ = center_B_0
        else:
            self.center_B_0_ = False

        if(adapt_in_features is not None):
            if(self.n_modes == 2):
                if(self.center_B_0_):
                    self.W = torch.nn.Sequential([torch.nn.Linear(in_features=adapt_in_features, out_features=1, bias=bias),
                                                torch.nn.Tanh()])
                else:
                    self.W = torch.nn.Sequential([torch.nn.Linear(in_features=adapt_in_features, out_features=1, bias=bias),
                                                torch.nn.Sigmoid()])
            
            else:
                self.W = torch.nn.Sequential([torch.nn.Linear(in_features=adapt_in_features, out_features=n_modes, bias=bias),
                                        torch.nn.Softmax(dim=1)])


        self.center_B_0_ = center_B_0;


    def forward(self, x : torch.Tensor, adapt_x : torch.Tensor):
        if(self.adapt_in_features is None):
            w_ = adapt_x
        else:
            w_ = self.W(adapt_x)
        b_ = self.B(x).view(x.shape[0], self.out_features, self.n_modes)

        if(self.n_modes == 2):
            if(self.center_B_0_ ):
                return b_[:,:,0] + b_[:,:,1]*w_
            else:
                return b_[:,:,0]*(1-w_) + b_[:,:,1]*w_
            
        else:
            return torch.matmul(b_, w_.view(x.shape[0], self.n_modes, 1)).squeeze(dim=2)
        
    def compute_adapt(self, adapt_x):
        with torch.no_grad():
            return self.W(adapt_x)
        

class QuickReducer(torch.nn.module):
    def __init__(self, M : torch.Tensor, normalize : bool = True):
        super().__init__()
        if(normalize):
            self.M      = M / torch.norm(M,dim=0,keepdim=True)
            self.s_bias = -torch.exp(0.5)
            self.s_norm = torch.e*(1-torch.e)
        else:
            self.M = M
            self.s_bias = 0
            self.s_norm = 1

    def forward(self, x):

        y = torch.matmul(x,self.M)
        return torch.hstack([y, (torch.square(y)-self.s_bias)/self.s_norm ])