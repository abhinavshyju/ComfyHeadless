import os
import sys

# Add the current directory to sys.path so comfy and nodes can be found
base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, base_dir)

import folder_paths
import nodes

def set_base_directory(path: str):
    """Sets the base directory for all ComfyUI folders (models, input, output, etc.)."""
    path = os.path.abspath(path)
    folder_paths.base_path = path
    folder_paths.models_dir = os.path.join(path, "models")
    folder_paths.output_directory = os.path.join(path, "output")
    folder_paths.input_directory = os.path.join(path, "input")
    folder_paths.temp_directory = os.path.join(path, "temp")
    
    # Update existing folder mappings
    for name in folder_paths.folder_names_and_paths:
        paths, extensions = folder_paths.folder_names_and_paths[name]
        # Only update paths that were relative to the old base_dir (models folder)
        new_paths = [os.path.join(folder_paths.models_dir, os.path.basename(p)) for p in paths]
        folder_paths.folder_names_and_paths[name] = (new_paths, extensions)

# Access all nodes via nodes.NODE_CLASS_MAPPINGS
# Example:
# UNETLoader = nodes.NODE_CLASS_MAPPINGS["UNETLoader"]()

def get_node_instance(node_class_name):
    if node_class_name in nodes.NODE_CLASS_MAPPINGS:
        return nodes.NODE_CLASS_MAPPINGS[node_class_name]()
    raise KeyError(f"Node class {node_class_name} not found in nodes.py")

if __name__ == "__main__":
    print(f"Headless ComfyUI initialized with {len(nodes.NODE_CLASS_MAPPINGS)} nodes.")
    # Example usage:
    # set_base_directory("/path/to/your/models")
    # unet_loader = get_node_instance("UNETLoader")
    # print(f"Successfully instantiated {unet_loader.__class__.__name__}")
