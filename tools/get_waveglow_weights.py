"""
Code to save weights from checkpoint in the form, which can be directly used by waveglow
"""

import os, sys
from scipy.io.wavfile import write
import torch
import numpy as np
from numpy.linalg import inv

# waveglow_path = "waveglow_256channels.pt"
waveglow_path = sys.argv[1]

waveglow = torch.load(waveglow_path)['model']
waveglow = waveglow.remove_weightnorm(waveglow)

folder = "waveglow_weights/"
for child in waveglow.named_children():
    # print("child is ", child)
    layer_name = child[0]
    layer_params = {}
    for param in child[1].named_parameters():
        print(param[0])
        param_name = param[0]
        param_value = param[1].data.cpu().numpy()
        np.save(folder +param_name.replace('.',"_"), param_value)
        layer_params[param_name] = param_value

for i in range(12):
    name = "{}_conv_weight.npy".format(i)
    a = np.load(folder+name)
    save_name = folder + "{}_conv_weight_inv.npy".format(i)
    # print(save_name, a.shape)
    a = np.expand_dims(inv(a.squeeze()),-1)
    print(name, a.shape)
    np.save(save_name, np.ascontiguousarray(a, dtype=np.float32))

