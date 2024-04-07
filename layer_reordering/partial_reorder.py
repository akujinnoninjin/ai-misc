#import torch
from safetensors.torch import load_file, save_file, save_model
import glob
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
import os
import json
# import yaml

model_folder = "../models/tinyllama-1.1b-chat-1.0"
# model_folder = "../models/mistral-7b-instruct-0.2"
file_pattern = "model*.safetensors"

file_list = glob.glob(os.path.join(model_folder, file_pattern))
total_files = len(file_list)
print(file_list)

model_dict = {}
for file_path in file_list:
    model_dict.update(load_file(file_path))

with open(os.path.join(model_folder, 'config.json')) as f:
    config_file =  json.load(f)
    f.close()

layers_base = int(config_file['num_hidden_layers'])
layers_io = layers_base//4
layers_mid = layers_base - layers_io * 2
assert(layers_mid > layers_io) # Sanity check; otherwise renaming end layers could cause overwrites

# Reverse sort the dict before iterating would guarantee avoiding the collisions

middle_layers = {}
# walk through all the keys
for key in list(model_dict.keys()):
    # Numbered layers
    if key.startswith('model.layers'):
        n = int(key.split(".")[2])
        
        # Rename end layers
        if n >= layers_base-layers_io:
            new_n = n+(layers_mid)
            new_key = key.replace(str(n), str(new_n))
            model_dict[new_key] = model_dict.pop(key)
            
        # Put mid layers in new object
        elif n >= layers_io:
            middle_layers[key] = model_dict.pop(key)


# print(middle_layers.keys())

layernorm_options = ["input_layernorm", "post_attention_layernorm"]

sort_by_input = False

if not sort_by_input:
    layernorm_options.reverse()

# create a list of the original layer numbers + sorting norms 
middle_sorting = []
for key in middle_layers.keys():
    if key.split(".")[3] == layernorm_options[0]:
        n = int(key.split(".")[2])
        mean = float(middle_layers[key].mean())
        middle_sorting.append([mean, n])

# print(middle_sorting)
middle_sorting.sort()
# print(middle_sorting)

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
