"""Visualise LLM layernorms on a graph
Usage: `attn_visualise.py [path]`

Where the optional `path` is either:
    - A single model (gguf)
    - A directory containing multiple models (gguf)
    - A directory containing a single model (safetensors, pytorch)
    
If no `path` is provided, the script will search the current directory.

Output graph is saved as a `.png` in the current directory, with a
filename based on the provided path. 
"""
__author__ = "AkaiStormherald"
__copyright__ = "Copyright 2024"
__status__ = "Development"
__license__ = "GNU GPLv3"

from pathlib import Path
from sys import stdout, argv, exit
import logging 

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from torch import float16, mean as tmean  
# Also requires gguf, safetensors

logger = logging.getLogger(__name__)

def main() -> int:
    logging.basicConfig(stream=stdout, level=logging.INFO)
    logger.debug(f'Command line arguments: {argv}')
    
    try:
        file_path = Path(argv[1])
    except:
        logger.info("Using current directory...")
        file_path = Path('.')
      
    file_list = get_filelist(file_path)
    if len(file_list) == 0:
        e = f"No model files found in {Path.cwd()}"
        logger.error(e)
        raise FileNotFoundError(e)
    
    model_dict = load_from_filelist(file_list)
    layernorms = get_layernorms(model_dict)
    del model_dict
    make_pretty_graph(layernorms, file_list[0])
    return 0
  
def get_filelist(file_path: Path) -> list:
    if file_path.is_dir():
        # Attempt to load all files from a directory
        file_extensions = [".safetensors", ".pytorch", ".gguf",]
        filename_patterns = ["model.*", "model-*-of-*.", "*.gguf",] 

        # Get all files with valid extensions
        file_list = [f for f in file_path.iterdir() 
                     if f.suffix.lower() in file_extensions]
        logger.debug(f"Files with valid extensions: {file_list}")
        
        # Then filter for valid name patterns
        file_list = [f for f in file_list
                     if any(f.match(fp) for fp in filename_patterns)]
        logger.debug(f"Files with valid names: {file_list}")
        return file_list
    
    elif file_path.is_file():
        # Handle single file
        file_list = [file_path]
        logger.debug(f"Single file found: {file_list}")
        return file_list
    
    else:
        e = f"{file_path} is not a file or directory"
        logger.error(e)
        raise IOError(e)

def load_from_filelist(file_list: list) -> dict:
    # Import the relevant file handlers as necessary
    match file_list[0].suffix.lower():
        case ".safetensors":
            logger.debug(f"Safetensors import based on {file_list[0]}")
            from safetensors.torch import load_file
        case ".pytorch":
            logger.debug(f"Pytorch import based on {file_list[0]}")
            from torch import load as load_file
        case ".gguf":
            logger.debug(f"GGUF import based on {file_list[0]}")
            from gguf.gguf_reader import GGUFReader
            def load_file(file: str) -> dict:
                gguf = GGUFReader(file, "r")
                gguf_dict = {}
                for tensor in gguf.tensors:
                    gguf_dict.update({tensor.name: tensor})
                return gguf_dict
        case _:
            e = f"Unsupported file: {file_list[0]}"
            logging.error(e)
            raise IOError(e)
    
    # Read the files into the layer dictionary
    model_dict = {}
    for i, file in enumerate(file_list):
        logger.info(f"Loading {file} ({i+1}/{len(file_list)})...")
        model_dict.update(load_file(file))
    
    # Quick sanity check we actually loaded anything
    assert(len(model_dict)>0)
    
    logger.info(f"Loading complete.")
    return model_dict

def get_layernorms(model_dict) -> list[dict,dict]:
    norms = [{}, {}] # Input, PostAttn
    
    def add_norm_to_dict(i: int, key: str, split: int, val) -> None:
        newkey = int(key.split(".")[split-1])
        logger.debug(f'Adding {key.split(".")[split]} - {newkey}')
        norms[i].update({newkey: val})
    
    for key,val in model_dict.items():  
        try: 
            # Is it pytorch/safetensors style?
            i = ["input_layernorm", "post_attention_layernorm"].index(key.split(".")[3])
            add_norm_to_dict(i, key, 3, val)
        except (ValueError, IndexError):
            try: 
                # Is it gguf style?
                i = ["ffn_norm", "attn_norm"].index(key.split(".")[2])
                add_norm_to_dict(i, key, 2, val)
            except (ValueError, IndexError):
                # Ignore it
                pass            
    return norms

def make_pretty_graph(norms: list[dict,dict], titlefile: Path) -> None:                
    if titlefile.suffix.lower() == ".gguf":
        title = titlefile.name
    else:
        title = titlefile.parent.name
    
    post_weights = []
    input_weights = []

    logger.info("Calculating means...")
    for i in range(0, len(norms[0])):
        input_weights.append(tmean(norms[0][i], dtype=float16))
        post_weights.append(tmean(norms[1][i], dtype=float16))

    logger.info("Generating graph...")
    # Create figure and axis objects
    _, ax = plt.subplots()

    # Plotting the two series
    ax.plot(range(len(input_weights)), input_weights, label='InputNorm', linewidth=2)
    ax.plot(range(len(post_weights)), post_weights, label='PostAtnNorm', linewidth=2)

    # Set labels and title
    ax.set_xlabel('Layer')
    ax.set_title(title)

    # Add vertical grid lines every 1 unit
    ax.grid(True, which='major', axis='x', linestyle='--', color='gray', linewidth=0.2)
    plt.grid(True, which='minor', axis='x', linestyle=':', color='gray', linewidth=0.2)

    plt.gca().xaxis.set_major_locator(MultipleLocator(4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

    # Add points on the line
    ax.scatter(range(len(input_weights)), input_weights, s=3)
    ax.scatter(range(len(post_weights)), post_weights, s=3)

    # Add horizontal line at 0
    plt.axhline(y=0, color="grey", linestyle="--", linewidth=1)

    # Add vertical marker lines at certain points
    #marker_points = [4, 11, 30, 37]  # Define the points where you want to add markers
    #for point in marker_points:
    #    plt.axvline(x=point, color='red', linestyle='--', linewidth=2)  # Adjust color, linestyle, and linewidth as needed
    
    # Show legend
    ax.legend()

    # Autoscale y axis
    ax.autoscale(axis='y')

    # Save the plot to an image file
    logging.info(f"Saving plot to {title}.png")
    # Todo: This should probably prompt to overwrite...
    plt.savefig(title+'.png')

    # Close the plot to avoid displaying it in GUI
    plt.close()

if __name__ == '__main__':
    exit(main())