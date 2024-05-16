"""Visualise LLM layernorms on a graph
Usage: `attn_vis.py [path]`

Where the optional `path` is either:
    - A single model file (safetensors, pytorch, gguf)
    - A directory containing:
        - One model (safetensors, pytorch, gguf)
        - Multiple models (gguf)
        - Subdirectories containing models (safetensors, pytorch, gguf)
            * Only searches one level deep
    
If no `path` is provided, the script will search the current directory.

Output graph is saved as a `.png` in the current directory, with a
filename based on the provided path. 
"""
__author__ = "AkaiStormherald"
__copyright__ = "Copyright 2024"
__status__ = "Development"
__license__ = "GNU GPLv3"

from pathlib import Path
import logging 
from sys import stdout, argv, exit

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# Also requires torch, gguf, safetensors

from icecream import ic

logger = logging.getLogger('attn_vis')
logging.basicConfig(stream=stdout, level=logging.INFO)

def main() -> int:
    logger.debug(f'Command line arguments: {argv}')
    try:
        file_path = Path(argv[1])
        if not file_path.exists():
            e = f"{file_path} not found."
            logger.error(e)
            raise FileNotFoundError(e)
    except IndexError:
        logger.info("Using current directory...")
        file_path = Path('.')
    
    try:
        attn_vis(file_path)
    except Exception as e:
        logger.error(f"Unhandled Exception: {e}")
        return 1
    
    return 0

def attn_vis(file_path: Path) -> None:
    """ Main attention visualiser routine
    """
    one_per_folder = [".safetensors", ".pytorch",]
    many_per_folder = [".gguf",]

    models = get_filelists(file_path, one_per_folder, many_per_folder)
    
    if models == None:
        e = f"No model files found in {Path.cwd()} or subdirectories..."
        logger.error(e)
        raise FileNotFoundError(e)
    
    for i, model in enumerate(models):
        # Make best guess at the model's name:
        f = Path(model[0])  
        if len(model) == 1:
            id = f"{f.parent.stem}\{f.name}"
        else:
            id = f"{f.parent.stem}"
        logger.info(f"Loading Model {i+1}/{len(models)} ({id}) ...")
        
        # Attempt to load, with cleanup 
        try:
            with ModelDict(model) as model_dict:
                if f.suffix.lower() in many_per_folder:
                    title = f.name
                else:
                    title = f.parent.stem
                make_pretty_graph(model_dict.inputnorm_means,
                                  model_dict.postattnnorm_means,
                                  title)
        except Exception as e:
            logger.error(f"Unhandled Error: {e}")
            raise(e)
        finally: 
            del model_dict

def get_filelists(file_path: Path, separated: list, loose: list) -> list:
    """ Return a list of paths filtered by extension, grouping separated items
    Only recurses one folder deep, see comment below to modify for unlimited
    
    Blanking on better ways to describe the two groupings -  
        separated: extensions for one-thing-per-folder
        loose: extensions for multiple-things-per-folder
    """

    # Get list of search folders (current and immediate subfolders)
    search_folders = [file_path]
    for subfolder in filter(Path.is_dir, file_path.iterdir()):
        # For unlimited recursion, replace
        #       file_path.iterdir()
        # with
        #       Path(file_path).glob('**/') 
        search_folders.append(subfolder)

    # Add the model files to the output list, grouping multiparts as one item
    outputs = []
    for search_folder in search_folders:
        for ext in loose:
            files = sorted(Path(search_folder).glob('*' + ext))
            for file in files:
                outputs.append([file]) 
        for ext in separated:
            files = sorted(Path(search_folder).glob('*' + ext))
            if files:
                outputs.append(files)
    return outputs

class ModelDict():
    """ ModelDict object class
    
    file_list: list of lists of model files
    inputnorm_means: list of inputnorm means
    postattnnorm_means: list of postattnnorm means
    """
    def __init__(self, file_list):
        self.file_list = file_list
        self.model_dict = {}
        
    def __enter__(self):
        self._load_from_filelist(self.file_list)
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        del self.model_dict
        logger.debug(f"Cleaned up model_dict")
    
    @property
    def inputnorm_means(self):
        if not hasattr(self, "_inputnorm_means"):
            self._calculate_layernorm_means()
        return self._inputnorm_means
    
    @property
    def postattnnorm_means(self):
        if not hasattr(self, "_postattnnorm_means"):
            self._postattnnorm_means()
        return self._postattnnorm_means
        
    def _load_from_filelist(self, file_list: list) -> dict:
        """ Loads a model from a file_list, returning a model_dict
        Late imports relevant file-handlers because they're huge and slow
        """
        # Import libraries as necessary, aliased as `load_file`
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
                from torch import from_numpy
                def load_file(file: str) -> dict:
                    gguf = GGUFReader(file, "r")
                    gguf_dict = {}
                    for tensor in gguf.tensors:
                        gguf_dict.update(
                            {tensor.name: from_numpy(tensor.data)})
                    return gguf_dict
            case _:
                e = f"Unsupported file: {file_list[0]}"
                logger.error(e)
                raise IOError(e)
    
        # Read the files into the layer dictionary
        for i, file in enumerate(file_list):
            logger.info(f"Reading part part {i+1}/{len(file_list)} ({file})...")
            try:
                self.model_dict.update(load_file(file))
            except Exception as e:
                logger.error(f"Something went wrong loading {file}: {e}")
                raise e
        logger.info(f"Loading complete.")

    def _calculate_layernorm_means(self) -> None:
        from torch import float16, mean as tmean
        attn_dict = {}
        ffn_dict = {}
        for layer, weights in self.model_dict.items():
            try:
                for x in [1,2]:
                    i = layer.split(".")[x]
                    if i.isnumeric():
                        break  
            except IndexError:
                pass 
            if not i.isnumeric():
                continue
            i = int(i)
              
            name = layer.split(".")[x+1]
            match name:
                case "input_layernorm" | "attn_norm":
                    attn_dict.update(
                        {i: tmean(weights, dtype=float16)})
                case "post_attention_layernorm" | "ffn_norm":
                    ffn_dict.update(
                        {i: tmean(weights, dtype=float16)})
                case _:
                    pass  
        
        self._inputnorm_means = []
        self._postattnnorm_means = []
        for _, value in sorted(attn_dict.items()):
            self.inputnorm_means.append(value)
        for _, value in sorted(ffn_dict.items()):
            self.postattnnorm_means.append(value)

def make_pretty_graph(series1: list, series2: list, title: str) -> None:  
    """ Make the pretty graph
    """
    logger.info(f"Generating graph...")
    # Create figure and axis objects
    _, ax = plt.subplots()

    # Plotting the two series
    ax.plot(range(len(series1)), series1,
            label='InputNorm', linewidth=2)
    ax.plot(range(len(series2)), series2,
            label='PostAtnNorm', linewidth=2)

    # Set labels and title
    ax.set_xlabel('Layer')
    ax.set_title(title)

    # Add vertical grid lines every 1 unit
    ax.grid(True, which='major', axis='x', linestyle='--',
            color='gray', linewidth=0.2)
    plt.grid(True, which='minor', axis='x', linestyle=':',
             color='gray', linewidth=0.2)

    plt.gca().xaxis.set_major_locator(MultipleLocator(4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

    # Add points on the line
    ax.scatter(range(len(series1)), series1, s=3)
    ax.scatter(range(len(series2)), series2, s=3)

    # Add horizontal line at 0
    plt.axhline(y=0, color="grey", linestyle="--", linewidth=1)

    # Add vertical marker lines at certain points
    #marker_points = [4, 11, 30, 37]
    #for point in marker_points:
    #    plt.axvline(x=point, color='red', linestyle='--', linewidth=2)
    
    # Show legend
    ax.legend()

    # Autoscale y axis
    ax.autoscale(axis='y')

    # Save the plot to an image file
    logger.info(f"Saving plot to {title}.png")
    # Todo: This should probably prompt to overwrite...
    plt.savefig(title+'.png')

    # Close the plot to avoid displaying it in GUI
    plt.close()

if __name__ == '__main__':
    exit(main())