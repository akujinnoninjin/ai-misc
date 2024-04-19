import torch
from safetensors.torch import load_file
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

file_pattern = "model-*-of-*.*"

file_list = glob.glob(file_pattern)
total_files = len(file_list)

if(total_files) == 0:
  file_pattern = "model.*"
  file_list = glob.glob(file_pattern)
  total_files = len(file_list)
  assert(total_files>0)

model_dict = {}

for file_path in file_list:
  if file_path.split(".")[-1] == ".safetensors":
      model_dict.update(load_file(file_path))
  elif file_path.split(".")[-1] == ".pytorch":
    model_dict.update(torch.load(file_path))
  assert(len(model_dict)>0)

# print(model_dict.keys())

layer_numbers = [int(key.split(".")[2]) for key in model_dict.keys() if key.startswith("model.layers.")]
max_layer_number = max(layer_numbers)
# print("The largest numbered key is:", max_layer_number)

post_weights = []
input_weights = []
# deltas = []

# print('Post:')
#for i in range(0,max_layer_number+1):
  # print(torch.mean(model_dict['model.layers.' + str(i) + '.post_attention_layernorm.weight']).item())
  
#print()
#print('In:')
#for i in range(0,max_layer_number+1):
#  print(torch.mean(model_dict['model.layers.' + str(i) + '.input_layernorm.weight']).item())

for i in range(0,max_layer_number+1):
  post_weights.append(torch.mean(model_dict['model.layers.' + str(i) + '.post_attention_layernorm.weight']).item())
  input_weights.append(torch.mean(model_dict['model.layers.' + str(i) + '.input_layernorm.weight']).item())
#  deltas.append(input_weights[i]-post_weights[i])
  
# print(post_weights)
# print(input_weights)

# Create figure and axis objects
fig, ax = plt.subplots()

# Plotting the two series
ax.plot(range(len(input_weights)), input_weights, label='InputNorm', linewidth=2)
ax.plot(range(len(post_weights)), post_weights, label='PostAtnNorm', linewidth=2)
# ax.plot(range(len(deltas)), deltas, label='Delta', linewidth=2)

# Set labels and title
ax.set_xlabel('Layer')
ax.set_title(os.path.basename(os.getcwd()))

# Add vertical grid lines every 1 unit
ax.grid(True, which='major', axis='x', linestyle='--', color='gray', linewidth=0.2)
plt.grid(True, which='minor', axis='x', linestyle=':', color='gray', linewidth=0.2)

plt.gca().xaxis.set_major_locator(MultipleLocator(4))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

# Add points on the line
ax.scatter(range(len(input_weights)), input_weights, s=3)
ax.scatter(range(len(post_weights)), post_weights, s=3)

plt.axhline(y=0, color="grey", linestyle="--", linewidth=1)

# Add vertical marker lines at certain points
marker_points = [4, 11, 30, 37]  # Define the points where you want to add markers
#for point in marker_points:
#    plt.axvline(x=point, color='red', linestyle='--', linewidth=2)  # Adjust color, linestyle, and linewidth as needed
# Show legend
ax.legend()

# Autoscale y axis
ax.autoscale(axis='y')

# Save the plot to an image file
plt.savefig(os.path.basename(os.getcwd())+'.png')

# Close the plot to avoid displaying it in GUI
plt.close()
