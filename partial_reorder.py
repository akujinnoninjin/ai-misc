#import torch
from safetensors.torch import load_file, save_file, save_model
import glob
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
import os
# import yaml

# model_folder = "/models/tinyllama-1.1b-chat-1.0"
model_folder = "/models/mistral-7b-instruct-0.2"
file_pattern = "model*.safetensors"

file_list = glob.glob(os.path.join(model_folder, file_pattern))
total_files = len(file_list)
print(file_list)

model_dict = {}
for file_path in file_list:
    model_dict.update(load_file(file_path))

# Get layer count
# layercount = 0
# for key in model_dict.keys():
#     if key.startswith('model.layers'):
#         layercount = max(layercount, int(key.split(".")[2]))
# io_layers = layercount//4

# layer_norms = []
# layer_count = 0

# import numpy as np

# for key in model_dict.keys():
#     if key.startswith('model.layers') and len(key.split(".")) in range (5,7):
#         layer_number = int(key.split(".")[2])
#         layer_count = max(layer_count, layer_number)
#         layer_type = key.split(".")[3]
#         if layer_type == 'post_attention_layernorm': #'input_layernorm':
#             layer_norms.append([layer_number, float(model_dict[key].mean()), float(model_dict[key].max()),
#                                 float(model_dict[key].min()), float(model_dict[key].float().quantile(0.005)), float(model_dict[key].float().quantile(0.995))])


#         # layer_subtype = key.split(".")[-2]
#         # print(f"{layer_type} {layer_subtype}")

# plot_min = []
# plot_max = []
# plot_avg = []
# plot_q25 = []
# plot_q75 = []

# layer_norms.sort()
# for ln in layer_norms:
#     #print(f"{ln[0]:2}. {ln[1]:1.3f}, {ln[2]:1.3f}, {ln[3]:1.3f}")
#     plot_min.append(ln[1]-ln[3])
#     plot_max.append(ln[2]-ln[1])
#     plot_avg.append(ln[1])
#     plot_q25.append(ln[4])
#     plot_q75.append(ln[5])
# # print(layer_norms[0:3][:])

# plot_errors = [plot_min, plot_max] 

# import matplotlib.pyplot as plt

# plt.errorbar(range(0,len(plot_avg)), plot_avg, yerr=plot_errors, elinewidth=0.5)
# plt.scatter(range(0,len(plot_avg)), plot_q25,s=4, c="#770000")
# plt.scatter(range(0,len(plot_avg)), plot_q75, s=4, c="#007700")
# plt.show()

### 3d version lol
layer_norms = []
layer_count = 0

import numpy as np


for key in model_dict.keys():
    if key.startswith('model.layers') and key.endswith('.post_attention_layernorm.weight'):
        layer_number = int(key.split(".")[2])
        layer_count = max(layer_count, layer_number)
#         layer_type = key.split(".")[3]
#         if layer_type == 'post_attention_layernorm': #'input_layernorm':
#             layer_norms.append([layer_number, float(model_dict[key].mean()), float(model_dict[key].max()),
#                                 float(model_dict[key].min()), float(model_dict[key].float().quantile(0.005)), float(model_dict[key].float().quantile(0.995))])


#         # layer_subtype = key.split(".")[-2]
#         # print(f"{layer_type} {layer_subtype}")

import matplotlib.pyplot as plt
from matplotlib import cm

ax = plt.axes(projection="3d")
xstep = np.arange(0,layer_count, 1)
#ystep = np.arange(0,1,1/2048)
ystep = np.arange(0,4096,1)

z = [[0 for i in range(len(ystep))] for j in range(0,len(xstep))]


#which_norm = "post_attention_layernorm"
which_norm = "input_layernorm"

for x in range(len(xstep)):
    for y in range(len(ystep)):
        z[x][y] = float(model_dict['model.layers.' + str(xstep[x]) + '.' + which_norm + '.weight'][y]) #.float().quantile(ystep[y]))

Z = np.array(z)
print(Z.shape)

X, Y = np.meshgrid(xstep,ystep, indexing='ij')

ax.plot_surface(X,Y,Z, antialiased=True, cmap=cm.plasma, rstride = 1, cstride = 1, edgecolor='darkred', linewidth=0.2)
ax.set_xlabel('Layer')
ax.set_ylabel('param#')
plt.show()


# plot_min = []
# plot_max = []
# plot_avg = []
# plot_q25 = []
# plot_q75 = []

# layer_norms.sort()
# for ln in layer_norms:
#     #print(f"{ln[0]:2}. {ln[1]:1.3f}, {ln[2]:1.3f}, {ln[3]:1.3f}")
#     plot_min.append(ln[1]-ln[3])
#     plot_max.append(ln[2]-ln[1])
#     plot_avg.append(ln[1])
#     plot_q25.append(ln[4])
#     plot_q75.append(ln[5])
# # print(layer_norms[0:3][:])

# plot_errors = [plot_min, plot_max] 

# import matplotlib.pyplot as plt

# plt.errorbar(range(0,len(plot_avg)), plot_avg, yerr=plot_errors, elinewidth=0.5)
# plt.scatter(range(0,len(plot_avg)), plot_q25,s=4, c="#770000")
# plt.scatter(range(0,len(plot_avg)), plot_q75, s=4, c="#007700")
# plt.show()


# input_layernorm <- sort by me
# post_attention_layernorm 
# mlp.down_proj
# mlp.up_proj
# mlp.gate_proj
# self_attn.k_proj
# self_attn.q_proj
# self_attn.v_proj
# self_attn.o_proj


# for key in model_dict.keys():
#     # if it's a layer key, with an expected name format
#     # ie. model.layers.[layernumber].[layertype](.layersubtype).weights
#     if key.startswith('model.layers') and len(key.split(".")) in range (5,7):
#         layer_number = int(key.split(".")[2])
#         layer_type = key.split(".")[3]
#         layer_subtype = key.split(".")[-2]

#         if layer_type != layer_subtype:
#             layer_type += ('.' + layer_subtype)

#         if layer_number in layers:
#             layers[layer_number].append(layer_type)
#         else:
#             layers[layer_number] = [layer_type]
#     else:
#         print(key)


# save_file(model_dict, "newmodel.safetensors")
