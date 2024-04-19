import torch
from safetensors.torch import load_file
import glob
import matplotlib.pyplot as plt
import os
from gguf.gguf_reader import GGUFReader, ReaderTensor
import numpy as np

#file_pattern = "model-*-of-*.safetensors"
file_pattern = "*.gguf"

file_list = glob.glob(file_pattern)

for file_name in file_list:
  try:
    reader = GGUFReader(file_name, "r")
  except:
    pass

  model_dict = {}

  for tensor in reader.tensors:
    model_dict.update({tensor.name: tensor})

# print(model_dict.keys())

  layer_numbers = [int(key.split(".")[1]) for key in model_dict.keys() if key.startswith("blk.")]
  max_layer_number = max(layer_numbers)
# print("The largest numbered key is:", max_layer_number)

  post_weights = []
  input_weights = []

  # print('Post:')
  #for i in range(0,max_layer_number+1):
    # print(torch.mean(model_dict['model.layers.' + str(i) + '.post_attention_layernorm.weight']).item())
    
  #print()
  #print('In:')
  #for i in range(0,max_layer_number+1):
  #  print(torch.mean(model_dict['model.layers.' + str(i) + '.input_layernorm.weight']).item())

  for i in range(0,max_layer_number+1):
    post_weights.append(np.mean(model_dict['blk.' + str(i) + '.ffn_norm.weight'].data).item())
    input_weights.append(np.mean(model_dict['blk.' + str(i) + '.attn_norm.weight'].data).item())
    
  # print(post_weights)
  # print(input_weights)

  # Create figure and axis objects
  fig, ax = plt.subplots()

  # Plotting the two series
  ax.plot(range(len(input_weights)), input_weights, label='Input')
  ax.plot(range(len(post_weights)), post_weights, label='PostAtn')

  # Set labels and title
  ax.set_xlabel('Layer')
  ax.set_title(str(file_name))

  # Show legend
  ax.legend()

  # Autoscale y axis
  ax.autoscale(axis='y')

  # Save the plot to an image file
  plt.savefig(str(file_name)+'.png')

# Close the plot to avoid displaying it in GUI
plt.close()
