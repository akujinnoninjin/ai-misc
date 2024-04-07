#import torch
from safetensors.torch import load_file, save_file, save_model
import glob
import os

# model_path = "/models/tinyllama-1.1b-chat-1.0"
model_path = "/models/mistral-7b-instruct-0.2"

def importModel(modelpath) -> dict:
    file_pattern = "model*.safetensors"

    file_list = glob.glob(os.path.join(model_path, file_pattern))
    total_files = len(file_list)
    print(file_list)

    model_dict = {}
    for file_path in file_list:
        model_dict.update(load_file(file_path))
        
    return model_dict

### 2D VERSION WITH ERROR BARS / PERCENTILES
# import matplotlib.pyplot as plt
# import numpy as np

# which_norm = 'post_attention_layernorm' # 'input_layernorm'
# quant = 0.005

# for key in model_dict.keys():
#     if key.startswith('model.layers') and len(key.split(".")) in range (5,7):
#         layer_number = int(key.split(".")[2])
#         layer_type = key.split(".")[3]
#         if layer_type == which_norm:
#             layer_norms.append([layer_number, float(model_dict[key].mean()), float(model_dict[key].max()),
#                                 float(model_dict[key].min()), float(model_dict[key].float().quantile(quant)),
#                                 float(model_dict[key].float().quantile(1-quant))])

# plot_min = []
# plot_max = []
# plot_avg = []
# plot_qdn = []
# plot_qup = []

# layer_norms.sort()
# for ln in layer_norms:
#     #print(f"{ln[0]:2}. {ln[1]:1.3f}, {ln[2]:1.3f}, {ln[3]:1.3f}")
#     plot_min.append(ln[1]-ln[3])
#     plot_max.append(ln[2]-ln[1])
#     plot_avg.append(ln[1])
#     plot_qdn.append(ln[4])
#     plot_qup.append(ln[5])
# # print(layer_norms[0:3][:])

# plot_errors = [plot_min, plot_max] 

# plt.errorbar(range(0,len(plot_avg)), plot_avg, yerr=plot_errors, elinewidth=0.5)
# plt.scatter(range(0,len(plot_avg)), plot_qdn, s=4, c="#770000")
# plt.scatter(range(0,len(plot_avg)), plot_qup, s=4, c="#007700")
# plt.show()


#### NEW 3D Version
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

model_dict = importModel(model_path)

# Get numbered layer count
layer_count = 0
for key in model_dict.keys():
    if key.startswith('model.layers'):
        layer_number = int(key.split(".")[2])
        layer_count = max(layer_count, layer_number)

# Set up graph and axis
ax = plt.axes(projection="3d")
xstep = np.arange(0,layer_count, 1)

# By param number:
ystep = np.arange(0,4096,1) 
# By quantile (sorted):
# ystep = np.arange(0,1,1/4096)

z = [[0 for i in range(len(ystep))] for j in range(0,len(xstep))]

# Select which norm
which_norm = "input_layernorm" # "post_attention_layernorm"

for x in range(len(xstep)):
    for y in range(len(ystep)):
        # By param number:
        z[x][y] = float(model_dict['model.layers.' + str(xstep[x]) + '.' + which_norm + '.weight'][y])
        # By quantile (sorted):
        # z[x][y] = float(model_dict['model.layers.' + str(xstep[x]) + '.' + which_norm + '.weight'].float().quantile(ystep[y]))

# Set up the arrays for matplot
Z = np.array(z)
X, Y = np.meshgrid(xstep,ystep, indexing='ij')

# Plot the axis
ax.plot_surface(X,Y,Z, antialiased=True, cmap=cm.plasma, rstride = 1, cstride = 1, edgecolor='darkred', linewidth=0.2)
ax.set_xlabel('Layer')
ax.set_ylabel('param#')
plt.show()
