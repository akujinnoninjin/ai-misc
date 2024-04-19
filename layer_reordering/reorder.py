import torch
from safetensors.torch import load_file
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import yaml

current_folder_name = os.path.basename(os.getcwd())

file_pattern = "model-*-of-*.safetensors"

file_list = glob.glob(file_pattern)
total_files = len(file_list)

model_dict = {}

for file_path in file_list:
    model_dict.update(load_file(file_path))
# print(model_dict.keys())

layer_numbers = [int(key.split(".")[2]) for key in model_dict.keys() if key.startswith("model.layers.")]
max_layer_number = max(layer_numbers)
# print("The largest numbered key is:", max_layer_number)

unsorted_layers = []

for i in range(0,max_layer_number+1):
  unsorted_layers.append((i, torch.mean(model_dict['model.layers.' + str(i) + '.post_attention_layernorm.weight']).item(), torch.mean(model_dict['model.layers.' + str(i) + '.input_layernorm.weight']).item()))
  
# print(unsorted_layers)

## this nect section tries to calculate the end layers
#s based on when the patn layer starts decreasing
## but it mught be better to hardcode or pass in as arg

# only look at last 25%
#start_index#  = len(unsorted_layers) - len(unsorted_layers) // 4
# last_quarter = unsorted_layers[start_index:]
# max_tuple = max(last_quarter, key=lambda x: x[1])

# print("Tuple with highest second value in the last quarter:", max_tuple)

# hardcoding this for now
input_layer_cutoff = 8
output_layer_cutoff = len(unsorted_layers) - 8

# sort the middle layers
middle_layers = unsorted_layers[input_layer_cutoff:output_layer_cutoff]
sorted_middle_layers = sorted(middle_layers, key=lambda x: x[1])

sorted_layers = unsorted_layers.copy()
sorted_layers[input_layer_cutoff:output_layer_cutoff] = sorted_middle_layers

# print("Updated value pairs:", sorted_layers)

# turn that shit into a mergekit yaml
output_dict = {
  'slices': [],
  'merge_method': 'passthrough',
  'dtype': 'float16'
}

# Add Model and Layer-range for each value pair
for sorted_layer in sorted_layers:
    model_info = {
        'model': 'outputs/' + str(current_folder_name),
        'layer_range': [sorted_layer[0],sorted_layer[0]+1]
    }
    output_dict['slices'].append({'sources': [model_info]})

# Output the dictionary to a YAML file
with open(current_folder_name + '-sorted.yaml', 'w') as yaml_file:
    yaml.dump(output_dict, yaml_file, default_flow_style=False)

# Get weights for graph 
before_weights = [unsorted_layer[1] for unsorted_layer in unsorted_layers]
after_weights = [sorted_layer[1] for sorted_layer in sorted_layers]
before_inputs = [unsorted_layer[2] for unsorted_layer in unsorted_layers]
after_inputs = [sorted_layer[2] for sorted_layer in sorted_layers]

#print(unsorted_layers)
#print(before_weights)

# Create figure and axis objects
fig, ax = plt.subplots(1,2, figsize=(10,5))

# Plotting the two series
ax[0].plot(range(len(before_weights)), before_weights, label='PostAtnNorm Before')
ax[0].plot(range(len(after_weights)), after_weights, label='PostAtnNorm After')

# Plotting the two series
ax[1].plot(range(len(before_inputs)), before_inputs, label='InputNorm Before')
ax[1].plot(range(len(after_inputs)), after_inputs, label='InputNorm After')

for i in range(0,2):
  ax[i].grid(True, which='major', axis='x', linestyle='--', color='gray', linewidth=0.2)
  ax[i].grid(True, which='minor', axis='x', linestyle=':', color='gray', linewidth=0.2)
  ax[i].xaxis.set_major_locator(MultipleLocator(4))
  ax[i].xaxis.set_minor_locator(MultipleLocator(1))
  
  # Set labels and title
  ax[i].set_xlabel('Layer')
  
  # Autoscale y axis
  ax[i].autoscale(axis='y')
  
  # Show legend
  ax[i].legend()

plt.suptitle(os.path.basename(os.getcwd()) + '-sorted(Plan)')

plt.tight_layout()

# Save the plot to an image file
plt.savefig(os.path.basename(os.getcwd())+'-sorted-plan.png', bbox_inches='tight', pad_inches=0.1)

# Close the plot to avoid displaying it in GUI
plt.close()