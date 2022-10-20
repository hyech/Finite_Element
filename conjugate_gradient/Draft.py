import json
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import torch.nn as nn

infile = open('FEM_converted.json', 'r')
var_dict = json.load(infile)
infile.close()

xcoord = torch.tensor([i[0] for i in var_dict["coord"]], dtype=float)

print(xcoord)