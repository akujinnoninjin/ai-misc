#import torch
from safetensors.torch import load_file, save_file, save_model
import glob
import os
import json

# model_folder = "/projects/ai/models/tinyllama-1.1b-chat-1.0"
# model_folder = "/projects/ai/models/mistral-7b-instruct-0.2"
model_folder = "./ai/models/upstage_SOLAR-10.7B-v1.0"
# model_folder = "."

layers_io = 8 # layers_count//4
sort_norm = "post_attention_layernorm" # input_layernorm
sort_targets = ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"]
# input_layernorm, post_attention_layernorm, mlp, self_attn

DRY_RUN = True

# Layer Object for the Numbered Layers
class NumberedLayer:
    def __init__(self, n :int, type :str, weights, subtype :str = None):
        self.n = self.orign = n
        self.layer_type = type
        self.subtype = subtype if subtype != type else None
        self.weights = weights
        
    def getkey(self) -> str:
        full_type = self.layer_type
        if self.subtype:
            full_type += "." + self.subtype
        key = ".".join(['model.layers', str(self.n), full_type, 'weight'])
        return key

# Get the model files
file_pattern = "model*.safetensors"
file_list = glob.glob(os.path.join(model_folder, file_pattern))
total_files = len(file_list)
print(file_list)

# Load the model files
model_dict = {}
for file_path in file_list:
    model_dict.update(load_file(file_path))
    
# Get the config file, to extract the layer count
with open(os.path.join(model_folder, 'config.json')) as f:
    config_file =  json.load(f)
    f.close()
del f

# Save the original key count for an assert sanity check near the end
original_key_count = len(model_dict.keys())

layers_count = int(config_file['num_hidden_layers'])
layers_mid = layers_count - layers_io * 2
assert(layers_mid > layers_io) # Obsolete sanity check

# Extract the numbered layers, generate sort order
layers = []
sort_helper = []
for key in list(model_dict.keys()):
    # Numbered layers
    if key.startswith('model.layers'):
        n = int(key.split(".")[2])
        # If this is the sorting tensor
        if (key.split(".")[3] == sort_norm 
                # And its in the middle / sort range
                and n in range(layers_io, layers_io+layers_mid)):
            # Then add the relevant info to the sorting helper
            sort_helper.append((float(model_dict[key].mean()),n))
        
        # Then pop the key into a layer object and add it to the list
        layers.append(NumberedLayer(n = n, 
                                    type = key.split(".")[3], 
                                    subtype = key.split(".")[-2], 
                                    weights = model_dict.pop(key) ))

# Sort the layer list by layer number, and the sort helper by the mean
layers.sort(key=lambda x: x.n)
sort_helper.sort()

# Apply the requested sort
for layer in layers:
    if layer.orign in range(layers_io, layers_io+layers_mid):
        if (layer.layer_type in sort_targets):
            layer.n = [y[1] for y in sort_helper].index(layer.n) + layers_io
            print(f"{layer.n} + {layer.orign} + {layer.layer_type}")
del sort_helper

# Sort layers again for debugging only
if DRY_RUN:
    layers.sort(key=lambda x: x.n)

# Add the layers back to the model dictionary
for layer in layers:
    model_dict.update( {layer.getkey() : layer.weights } )
    if DRY_RUN:
        print(f"{layer.n} <- {layer.orign} {layer.layer_type}.{layer.subtype}")
del layers
del layer

# Confirm the new dict is the same size as the old dict
new_key_count = len(model_dict.keys())
assert(original_key_count == new_key_count)

# Save the output, or print the debug
if DRY_RUN:
    print(f"{original_key_count} == {new_key_count}?")
else:
    save_file(model_dict, "model.safetensors")

# input_layernorm
# post_attention_layernorm  <- sort by me
# mlp.down_proj
# mlp.up_proj
# mlp.gate_proj
# self_attn.k_proj
# self_attn.q_proj
# self_attn.v_proj
# self_attn.o_proj
