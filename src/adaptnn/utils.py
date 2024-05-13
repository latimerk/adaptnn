from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy.typing as npt

'''
===============================================================================
===============================================================================
Utility functions
===============================================================================
===============================================================================
'''
def compute_conv_output_size(L_in : int, kernel_size : int, stride : int = 1,
                             dilation : int=1, padding : int = 0) -> int:
    L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1
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